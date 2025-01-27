import torch
import os
import torch.nn as nn
import numpy as np
from collections import namedtuple
from huggingface_hub import hf_hub_download
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from nnsight import LanguageModel
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.buffer import ActivationBuffer
import argparse





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

# modify the following subclass to implement the proposed SAE variant
# change the name "CustomSAE" to a appropriate name such as "TemporalSAE" depending on experiment idea
class PositionalSAE(nn.Module):
    """Implementation of a Positional Sparse Autoencoder with position-specific masks."""
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        hook_layer: int,
        model_name: str = "EleutherAI/pythia-70m-deduped",
        hook_name: Optional[str] = None,
        context_length: int = 128,
    ):
        super().__init__()
        # Store dimensions as attributes first
        self.d_in = d_in
        self.d_sae = d_sae
        self.context_length = context_length
        self.activation_dim = d_in
        self.dict_size = d_sae
        
        # Initialize network parameters
        self.W_enc = nn.Parameter(torch.zeros(d_in, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        
        # Add layer normalization
        self.layer_norm = nn.LayerNorm(d_sae)
        
        # Initialize position-range-specific encoder weights with orthogonal initialization
        self.n_ranges = 4  # Split context into 4 ranges
        self.range_size = self.context_length // self.n_ranges
        self.W_enc_ranges = nn.ParameterList([
            nn.Parameter(torch.empty(d_in, d_sae // self.n_ranges))
            for _ in range(self.n_ranges)
        ])
        for W in self.W_enc_ranges:
            nn.init.orthogonal_(W)
            
        # Initialize position embeddings for each range
        self.pos_embeddings = nn.Parameter(torch.randn(self.n_ranges, d_in) * 0.02)
        
        # Shared decoder with orthogonal initialization
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
        nn.init.orthogonal_(self.W_dec)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32

        # Add CustomSAEConfig integration
        if hook_name is None:
            hook_name = f"blocks.{hook_layer}.hook_resid_post"

        self.cfg = CustomSAEConfig(
            model_name=model_name,
            d_in=d_in,
            d_sae=d_sae,
            hook_name=hook_name,
            hook_layer=hook_layer,
            architecture="PositionalSAE",
            activation_fn_str="relu",
            apply_b_dec_to_input=True,
            context_size=context_length,
        )

    def _get_position_range(self, pos_indices):
        """Get range index for each position."""
        return pos_indices // self.range_size

    def encode(self, input_acts):
        # Get sequence position indices
        batch_size = input_acts.shape[0]
        pos_indices = torch.arange(min(batch_size, self.context_length), device=input_acts.device)
        range_indices = self._get_position_range(pos_indices)
        
        # Initialize output tensor
        encoded = torch.zeros(batch_size, self.d_sae, device=input_acts.device)
        
        # Process each position range separately
        for range_idx in range(self.n_ranges):
            range_mask = (range_indices == range_idx)
            if not range_mask.any():
                continue
                
            range_input = input_acts[range_mask]
            # Add position embedding for this range
            range_input = range_input + self.pos_embeddings[range_idx]
            
            start_idx = range_idx * (self.d_sae // self.n_ranges)
            end_idx = (range_idx + 1) * (self.d_sae // self.n_ranges)
            
            # Compute activations for this range with layer norm before ReLU
            pre_acts = (range_input - self.b_dec) @ self.W_enc_ranges[range_idx] + self.b_enc[start_idx:end_idx]
            normalized_pre_acts = self.layer_norm(pre_acts)
            acts = torch.relu(normalized_pre_acts)
            
            # Place in output tensor
            encoded[range_mask, start_idx:end_idx] = acts
            
        return encoded, encoded  # Return same tensor twice for compatibility

    def decode(self, acts):
        return (acts @ self.W_dec) + self.b_dec

    def forward(self, acts, output_features=False):
        masked_encoded, raw_encoded = self.encode(acts)
        decoded = self.decode(masked_encoded)
        if output_features:
            return decoded, (masked_encoded, raw_encoded)
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

# modify the following subclass to implement the proposed SAE variant training
# change the name "CustomTrainer" to a appropriate name to match the SAE class name.
class PositionalTrainer(SAETrainer):
    """Trainer for Positional Sparse Autoencoder using L1 regularization and position-specific masking."""
    def __init__(self,
                 activation_dim=512,
                 dict_size=64*512,
                 lr=1e-3, 
                 l1_penalty=1e-1,
                 warmup_steps=1000,
                 resample_steps=None,
                 seed=None,
                 device=None,
                 layer=None,
                 lm_name=None,
                 wandb_name='CustomTrainer',
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

        # Initialize autoencoder
        self.ae = PositionalSAE(
            d_in=activation_dim, 
            d_sae=dict_size, 
            hook_layer=layer, 
            model_name=lm_name,
            context_length=128
        )

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

        # Initialize optimizer with constrained decoder weights and gradient clipping
        self.optimizer = ConstrainedAdam(
            self.ae.parameters(),
            [self.ae.W_dec],  # Constrain decoder weights
            lr=lr/10  # Reduce learning rate for stability
        )
        # Add gradient clipping
        self.max_grad_norm = 1.0
        
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
        x_hat, (masked_f, raw_f) = self.ae(x, output_features=True)
        l2_loss = torch.linalg.norm(x - x_hat, dim=-1).mean()
        
        # Apply L1 penalty on raw activations before masking
        l1_loss = raw_f.norm(p=1, dim=-1).mean()

        if self.steps_since_active is not None:
            # Update steps_since_active using masked activations
            deads = (masked_f == 0).all(dim=0)
            self.steps_since_active[deads] += 1
            self.steps_since_active[~deads] = 0
        
        loss = l2_loss + self.l1_penalty * l1_loss

        if not logging:
            return loss
        else:
            return namedtuple('LossLog', ['x', 'x_hat', 'masked_f', 'raw_f', 'losses'])(
                x, x_hat, masked_f, raw_f,
                {
                    'l2_loss': l2_loss.item(),
                    'mse_loss': (x - x_hat).pow(2).sum(dim=-1).mean().item(),
                    'sparsity_loss': l1_loss.item(),
                    'loss': loss.item()
                }
            )

    def update(self, step, activations):
        activations = activations.to(self.device)

        # Update mask coefficient based on training progress
        with torch.no_grad():
            progress = min(step / (self.warmup_steps * 2), 1.0)  # Slower progression
            self.ae.mask_coefficient.fill_(progress * 6.0 - 3.0)  # Sigmoid range

        self.optimizer.zero_grad()
        loss = self.loss(activations)
        loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.ae.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()

        if self.resample_steps is not None and step % self.resample_steps == 0:
            self.resample_neurons(self.steps_since_active > self.resample_steps / 2, activations)

    @property
    def config(self):
        return {
            'trainer_class': 'CustomTrainer',
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
    model_name: str = "google/gemma-2b",
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

    # Initialize model and buffer
    model = LanguageModel(
        model_name,
        device_map=device,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        cache_dir=None,
    )
    # added for pythia-70m
    if model_name == "EleutherAI/pythia-70m-deduped":
        # Access the transformer layers directly from the model
        submodule = model.gpt_neox.layers[layer]
    else:
        submodule = model.model.layers[layer]
    submodule_name = f"resid_post_layer_{layer}"
    activation_dim = model.config.hidden_size

    # Setup dataset and buffer
    generator = hf_dataset_to_generator("monology/pile-uncopyrighted")
    activation_buffer = ActivationBuffer(
        generator,
        model,
        submodule,
        n_ctxs=buffer_size,
        ctx_len=context_length,
        refresh_batch_size=llm_batch_size,
        out_batch_size=sae_batch_size,
        io="out",
        d_submodule=activation_dim,
        device=device,
    )

    # Initialize trainer  
    trainer = PositionalTrainer(
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
    for step in range(steps):
        activations = next(activation_buffer)
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
        "config": trainer.config,
        "final_info": final_info
    }

    # Save results using numpy format (similar to mech_interp)
    with open(os.path.join(out_dir, "all_results.npy"), "wb") as f:
        np.save(f, results)

    all_info_path = os.path.join(out_dir, "final_info.json")
    if os.path.exists(all_info_path):
        with open(all_info_path, 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = {}
    existing_data.update(final_info)
    with open(all_info_path, "w") as f:
        json.dump(existing_data, indent=2, fp=f)   
    return trainer.ae

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Any, Optional, List, Dict, Union, Tuple
from tqdm import tqdm

RANDOM_SEED = 42




MODEL_CONFIGS = {
    # "EleutherAI/pythia-70m-deduped": {"batch_size": 512, "dtype": "float32", "layers": [3, 4], "d_model": 512},
    "google/gemma-2-2b": {"batch_size": 32, "dtype": "bfloat16", "layers": [5, 12, 19], "d_model": 2304},
}



def str_to_dtype(dtype_str: str) -> torch.dtype:
    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(dtype_str.lower())
    if dtype is None:
        raise ValueError(
            f"Unsupported dtype: {dtype_str}. Supported dtypes: {list(dtype_map.keys())}"
        )
    return dtype

if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
    args = parser.parse_args()
    save_dir = args.out_dir
    
    
    # Do not modify this to gemma-2b models, gemma-2-2b is a different model and actually exists 
    model_name = "google/gemma-2-2b"
    # model_name = "EleutherAI/pythia-70m-deduped"
    d_model = MODEL_CONFIGS[model_name]["d_model"]
    llm_batch_size = MODEL_CONFIGS[model_name]["batch_size"]
    llm_dtype = MODEL_CONFIGS[model_name]["dtype"]
    # Initialize variables that were previously args
    layers = MODEL_CONFIGS[model_name]["layers"]
    num_tokens = 1000 # Set default number of tokens
    device = "cuda" if torch.cuda.is_available() else "cpu"
    no_wandb_logging = False # Set default wandb logging flag
    
    saes = []
    for layer in layers:
        sae = run_sae_training(
            layer=layer,
            dict_size=d_model,
            num_tokens=num_tokens,
            out_dir=save_dir,
            device=device,
            model_name=model_name,
            context_length=128,
            buffer_size=2048,
            llm_batch_size=llm_batch_size,
            sae_batch_size=2048,
            learning_rate=3e-4,
            sparsity_penalty=0.04,
            warmup_steps=1000,
            seed=42,
            wandb_logging=not no_wandb_logging,
            wandb_entity=None,
            wandb_project=None
        )
        saes.append(sae)
        
        # Save the trained model
        checkpoint = {
            "model_state_dict": sae.state_dict(),
            "layer": layer,
            "config": sae.cfg
        }
        save_path = os.path.join(save_dir, f"sae_layer_{layer}.pt")
        torch.save(checkpoint, save_path)
        print(f"Saved model checkpoint to {save_path}")
