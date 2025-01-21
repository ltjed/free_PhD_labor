import torch
import os
import torch.nn as nn
from collections import namedtuple
from huggingface_hub import hf_hub_download
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

# Do not modify CustomSAEConfig class as this defines the right format for SAE to be evaluated!
@dataclass
class CustomSAEConfig:
    model_name: str
    d_in: int
    d_sae: int
    hook_layer: int
    hook_name: str

    # The following are used for the core/main.py SAE evaluation
    context_size: int = None  # Can be used for auto-interp
    hook_head_index: Optional[int] = None

    # Architecture settings
    architecture: str = ""
    apply_b_dec_to_input: bool = None
    finetuning_scaling_factor: bool = None
    activation_fn_str: str = ""
    activation_fn_kwargs = {}
    prepend_bos: bool = True
    normalize_activations: str = "none"

    # Model settings
    dtype: str = ""
    device: str = ""
    model_from_pretrained_kwargs = {}

    # Dataset settings
    dataset_path: str = ""
    dataset_trust_remote_code: bool = True
    seqpos_slice: tuple = (None,)
    training_tokens: int = -100_000

    # Metadata
    sae_lens_training_version: Optional[str] = None
    neuronpedia_id: Optional[str] = None


class TemporalSAE(nn.Module):
    """An implementation of a Temporal Sparse Autoencoder with LSTM-based dynamics."""
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        hook_layer: int,
        window_size: int = 16,
        lstm_hidden_size: int = 256,
        model_name: str = "pythia-70m",
        hook_name: Optional[str] = None,
    ):
        super().__init__()
        
        # Static components
        self.W_enc = nn.Parameter(torch.zeros(d_in, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        
        # Temporal components
        self.window_size = window_size
        self.lstm_hidden_size = lstm_hidden_size
        
        # LSTM encoder/decoder
        self.lstm_encoder = nn.LSTM(d_sae, lstm_hidden_size, batch_first=True)
        self.lstm_decoder = nn.LSTM(lstm_hidden_size, d_sae, batch_first=True)
        
        # Sparse gates
        self.encoder_gate = nn.Sequential(
            nn.Linear(d_sae, d_sae),
            nn.Sigmoid()
        )
        self.decoder_gate = nn.Sequential(
            nn.Linear(d_sae, d_sae),
            nn.Sigmoid()
        )
        
        # Feature memory buffer
        self.register_buffer('feature_memory', torch.zeros(window_size, d_sae))
        self.register_buffer('feature_lifecycles', torch.zeros(d_sae))
        
        # Bridge connections
        self.static_temporal_bridge = nn.Linear(d_sae, lstm_hidden_size)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        
        # Add properties to match the interface expected by VanillaTrainer
        self.activation_dim = d_in
        self.dict_size = d_sae

        # Add CustomSAEConfig integration
        if hook_name is None:
            hook_name = f"blocks.{hook_layer}.hook_resid_post"

        self.cfg = CustomSAEConfig(
            model_name=model_name,
            d_in=d_in,
            d_sae=d_sae,
            hook_name=hook_name,
            hook_layer=hook_layer,
            # Set some reasonable defaults for VanillaSAE
            architecture="vanilla",
            activation_fn_str="relu",
            apply_b_dec_to_input=True,
        )

    def encode(self, input_acts):
        # Static encoding
        pre_acts = (input_acts - self.b_dec) @ self.W_enc + self.b_enc
        static_acts = torch.relu(pre_acts)
        
        # Apply sparse gate
        gate_values = self.encoder_gate(static_acts)
        gated_acts = static_acts * gate_values
        
        # Update feature memory
        self.feature_memory = torch.roll(self.feature_memory, -1, dims=0)
        self.feature_memory[-1] = gated_acts.mean(dim=0)
        
        # Temporal encoding
        memory_batch = self.feature_memory.unsqueeze(0).expand(input_acts.shape[0], -1, -1)
        temporal_features, _ = self.lstm_encoder(memory_batch)
        
        # Bridge connection
        bridge_features = self.static_temporal_bridge(gated_acts)
        
        return static_acts, temporal_features[:, -1], gate_values

    def decode(self, static_acts, temporal_features):
        # Combine static and temporal features
        gate_values = self.decoder_gate(static_acts)
        gated_static = static_acts * gate_values
        
        # Temporal decoding
        temporal_seq = temporal_features.unsqueeze(1)
        decoded_temporal, _ = self.lstm_decoder(temporal_seq)
        
        # Combine and decode
        combined_features = gated_static + decoded_temporal.squeeze(1)
        return (combined_features @ self.W_dec) + self.b_dec

    def forward(self, acts, output_features=False):
        static_encoded, temporal_encoded, gates = self.encode(acts)
        decoded = self.decode(static_encoded, temporal_encoded)
        
        # Update feature lifecycles
        active_features = (gates > 0.1).float()
        self.feature_lifecycles = self.feature_lifecycles * active_features.mean(dim=0)
        self.feature_lifecycles += active_features.mean(dim=0)
        
        if output_features:
            return decoded, static_encoded, temporal_encoded, gates
        return decoded

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device = kwargs.get("device", None)
        dtype = kwargs.get("dtype", None)
        if device:
            self.device = device
        if dtype:
            self.dtype = dtype
        return self


class ConstrainedAdam(torch.optim.Adam):
    """A variant of Adam where some parameters are constrained to have unit norm."""
    def __init__(self, params, constrained_params, lr):
        super().__init__(params, lr=lr)
        self.constrained_params = list(constrained_params)
    
    def step(self, closure=None):
        with torch.no_grad():
            for p in self.constrained_params:
                normed_p = p / p.norm(dim=0, keepdim=True)
                p.grad -= (p.grad * normed_p).sum(dim=0, keepdim=True) * normed_p
        super().step(closure=closure)
        with torch.no_grad():
            for p in self.constrained_params:
                p /= p.norm(dim=0, keepdim=True)


class SAETrainer:
    """Base class for implementing SAE training algorithms."""
    def __init__(self, seed=None):
        self.seed = seed
        self.logging_parameters = []

    def update(self, step, activations):
        """Update step for training. To be implemented by subclasses."""
        pass

    def get_logging_parameters(self):
        stats = {}
        for param in self.logging_parameters:
            if hasattr(self, param):
                stats[param] = getattr(self, param)
            else:
                print(f"Warning: {param} not found in {self}")
        return stats
    
    @property
    def config(self):
        return {
            'wandb_name': 'trainer',
        }


class TemporalTrainer(SAETrainer):
    """Trainer for Temporal Sparse Autoencoder using L1 regularization and temporal consistency."""
    def __init__(self,
                 activation_dim=512,
                 dict_size=64*512,
                 lr=1e-3,
                 l1_penalty=1e-1,
                 temporal_consistency_weight=0.1,
                 warmup_steps=1000,
                 resample_steps=None,
                 seed=None,
                 device=None,
                 layer=None,
                 lm_name=None,
                 wandb_name='TemporalTrainer',
                 submodule_name=None,
    ):
        super().__init__(seed)

        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Initialize temporal autoencoder
        self.ae = TemporalSAE(d_in=activation_dim, d_sae=dict_size)
        self.temporal_consistency_weight = temporal_consistency_weight

        self.lr = lr
        self.l1_penalty = l1_penalty
        self.warmup_steps = warmup_steps
        self.wandb_name = wandb_name

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.ae.to(self.device)

        self.resample_steps = resample_steps

        if self.resample_steps is not None:
            self.steps_since_active = torch.zeros(self.ae.dict_size, dtype=int).to(self.device)
        else:
            self.steps_since_active = None 

        # Initialize optimizer with constrained decoder weights
        self.optimizer = ConstrainedAdam(
            self.ae.parameters(),
            [self.ae.W_dec],  # Constrain decoder weights
            lr=lr
        )
        
        # Setup learning rate warmup
        if resample_steps is None:
            def warmup_fn(step):
                return min(step / warmup_steps, 1.)
        else:
            def warmup_fn(step):
                return min((step % resample_steps) / warmup_steps, 1.)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=warmup_fn)

    def resample_neurons(self, deads, activations):
        with torch.no_grad():
            if deads.sum() == 0:
                return
            print(f"resampling {deads.sum().item()} neurons")

            # Compute loss for each activation
            losses = (activations - self.ae(activations)).norm(dim=-1)

            # Sample input to create encoder/decoder weights from
            n_resample = min([deads.sum(), losses.shape[0]])
            indices = torch.multinomial(losses, num_samples=n_resample, replacement=False)
            sampled_vecs = activations[indices]

            # Get norm of the living neurons
            alive_norm = self.ae.W_enc[~deads].norm(dim=-1).mean()

            # Resample first n_resample dead neurons
            deads[deads.nonzero()[n_resample:]] = False
            self.ae.W_enc[deads] = sampled_vecs * alive_norm * 0.2
            self.ae.W_dec[:,deads] = (sampled_vecs / sampled_vecs.norm(dim=-1, keepdim=True)).T
            self.ae.b_enc[deads] = 0.

            # Reset Adam parameters for dead neurons
            state_dict = self.optimizer.state_dict()['state']
            for param_id, param in enumerate(self.optimizer.param_groups[0]['params']):
                if param_id == 0:  # W_enc
                    state_dict[param]['exp_avg'][deads] = 0.
                    state_dict[param]['exp_avg_sq'][deads] = 0.
                elif param_id == 1:  # W_dec
                    state_dict[param]['exp_avg'][:,deads] = 0.
                    state_dict[param]['exp_avg_sq'][:,deads] = 0.
                elif param_id == 2:  # b_enc
                    state_dict[param]['exp_avg'][deads] = 0.
                    state_dict[param]['exp_avg_sq'][deads] = 0.
    
    def loss(self, x, logging=False, **kwargs):
        x_hat, static_f, temporal_f, gates = self.ae(x, output_features=True)
        
        # Reconstruction loss
        l2_loss = torch.linalg.norm(x - x_hat, dim=-1).mean()
        
        # Sparsity losses
        l1_loss = static_f.norm(p=1, dim=-1).mean()
        gate_sparsity = gates.mean()
        
        # Temporal consistency loss
        temporal_consistency = torch.linalg.norm(
            temporal_f[1:] - temporal_f[:-1], dim=-1
        ).mean() if temporal_f.size(0) > 1 else torch.tensor(0.0).to(x.device)

        if self.steps_since_active is not None:
            # Update steps_since_active
            deads = (static_f == 0).all(dim=0)
            self.steps_since_active[deads] += 1
            self.steps_since_active[~deads] = 0
        
        # Combined loss
        loss = (
            l2_loss +
            self.l1_penalty * l1_loss +
            self.temporal_consistency_weight * temporal_consistency
        )

        if not logging:
            return loss
        else:
            return namedtuple('LossLog', ['x', 'x_hat', 'features', 'losses'])(
                x, x_hat, static_f,
                {
                    'l2_loss': l2_loss.item(),
                    'mse_loss': (x - x_hat).pow(2).sum(dim=-1).mean().item(),
                    'sparsity_loss': l1_loss.item(),
                    'loss': loss.item()
                }
            )

    def update(self, step, activations):
        activations = activations.to(self.device)

        self.optimizer.zero_grad()
        loss = self.loss(activations)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        if self.resample_steps is not None and step % self.resample_steps == 0:
            self.resample_neurons(self.steps_since_active > self.resample_steps / 2, activations)

    @property
    def config(self):
        return {
            'trainer_class': 'VanillaTrainer',
            'activation_dim': self.ae.activation_dim,
            'dict_size': self.ae.dict_size,
            'lr': self.lr,
            'l1_penalty': self.l1_penalty,
            'warmup_steps': self.warmup_steps,
            'resample_steps': self.resample_steps,
            'device': self.device,
            'layer': self.layer,
            'lm_name': self.lm_name,
            'wandb_name': self.wandb_name,
            'submodule_name': self.submodule_name,
        }


def run_sae_training(
    layer: int,
    dict_size: int,
    num_tokens: int,
    out_dir: str,  # Changed from save_dir to out_dir for consistency
    device: str,
    model_name: str = "facebook/opt-350m",
    context_length: int = 128,
    buffer_size: int = 2048,
    llm_batch_size: int = 24,
    sae_batch_size: int = 2048,
    learning_rate: float = 3e-4,
    sparsity_penalty: float = 0.04,
    warmup_steps: int = 1000,
    seed: int = 0,
    wandb_logging: bool = False,
    wandb_entity: str = None,
    wandb_project: str = None,
):
    # Convert out_dir to absolute path and create directory
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Calculate steps
    steps = int(num_tokens / sae_batch_size)

    # Initialize model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32,  # OPT models work better with float32
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Get activation dimension from model config
    activation_dim = model.config.hidden_size
    
    # Setup dataset
    dataset = load_dataset("monology/pile-uncopyrighted", streaming=True)
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=context_length,
            return_tensors="pt",
            padding="max_length"
        )
    
    # Create dataloader
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    dataloader = DataLoader(
        tokenized_dataset["train"],
        batch_size=sae_batch_size,
        shuffle=True
    )
    
    # Get layer module
    submodule = model.model.layers[layer]
    submodule_name = f"resid_post_layer_{layer}"

    # Initialize trainer  
    trainer = TemporalTrainer(
        activation_dim=activation_dim,
        dict_size=dict_size,
        lr=learning_rate,
        l1_penalty=sparsity_penalty,
        warmup_steps=warmup_steps,
        seed=seed,
        device=device,
        layer=layer,
        lm_name=model_name,
        submodule_name=submodule_name
    )

    training_log = []
    
    # Training loop
    training_iterator = iter(dataloader)
    for step in range(steps):
        try:
            batch = next(training_iterator)
        except StopIteration:
            training_iterator = iter(dataloader)
            batch = next(training_iterator)
            
        # Get activations from the model
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True)
            activations = outputs.hidden_states[layer]
            
        loss_dict = trainer.update(step, activations)
        training_log.append(loss_dict)
        
        if step % 100 == 0:
            print(f"Step {step}: {loss_dict}")
            
            if wandb_logging and wandb_entity and wandb_project:
                import wandb
                wandb.log(loss_dict, step=step)

    # Prepare final results
    final_info = {
        "training_steps": steps,
        "final_loss": training_log[-1]["loss"] if training_log else None,
        "layer": layer,
        "dict_size": dict_size,
        "learning_rate": learning_rate,
        "sparsity_penalty": sparsity_penalty
    }

    # Save model checkpoint 
    checkpoint = {
        "model_state_dict": trainer.ae.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
    }
    torch.save(checkpoint, os.path.join(out_dir, "autoencoder_checkpoint.pt"))

    # Save all results and metrics
    results = {
        "training_log": training_log,
        "config": trainer.config(),
        "final_info": final_info
    }

    # Save results using numpy format (similar to mech_interp)
    with open(os.path.join(out_dir, "all_results.npy"), "wb") as f:
        np.save(f, results)

    # Save final info separately as JSON (similar to mech_interp) 
    with open(os.path.join(out_dir, "final_info.json"), "w") as f:
        json.dump(final_info, f, indent=2)

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Optional

RANDOM_SEED = 42

def run(out_dir: str):
    """
    Run the SAE training experiment with the same directory structure as mech_interp.
    
    Args:
        out_dir: str, the output directory where results will be saved
    """
    out_dir = os.path.abspath(out_dir)
    
    # Create run_i directory structure
    i = 0
    while os.path.exists(os.path.join(out_dir, f"run_{i}")):
        i += 1
    run_dir = os.path.join(out_dir, f"run_{i}")
    os.makedirs(run_dir, exist_ok=True)

    # Run the training with default parameters
    run_sae_training(
        layer=5,  # Using first layer from gemma-2b config
        dict_size=512,  # Standard dictionary size
        num_tokens=1_000_000,  # 1M tokens for training
        out_dir=run_dir,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

if __name__ == "__main__":
    # Example usage
    run("./output")
