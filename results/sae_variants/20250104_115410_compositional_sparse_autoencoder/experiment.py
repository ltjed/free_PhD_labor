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


class CompSAE(nn.Module):
    """A Compositional Sparse Autoencoder with two-stream architecture and multi-head attention."""
    def __init__(
        self,
        d_in: int,
        d_f: int = 256,  # Feature extractor dimension
        d_c: int = 128,  # Composition network dimension
        n_heads: int = 8,
        hook_layer: int = 0,
        model_name: str = "facebook/opt-125m",
        hook_name: Optional[str] = None,
        bottleneck_k: int = 32,
    ):
        super().__init__()
        
        # Feature extractor stream
        self.W_enc_f = nn.Parameter(torch.zeros(d_in, d_f))
        self.b_enc_f = nn.Parameter(torch.zeros(d_f))
        
        # Composition network stream
        self.W_enc_c = nn.Parameter(torch.zeros(d_f, d_c))
        self.b_enc_c = nn.Parameter(torch.zeros(d_c))
        
        # Multi-head attention for feature composition
        self.n_heads = n_heads
        self.head_dim = d_c // n_heads
        self.mha = nn.MultiheadAttention(d_c, n_heads, batch_first=True)
        
        # Bottleneck layer
        self.bottleneck = nn.Parameter(torch.zeros(d_c, bottleneck_k))
        
        # Decoder
        self.W_dec = nn.Parameter(torch.zeros(bottleneck_k, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        
        # Residual connection
        self.W_res = nn.Parameter(torch.zeros(d_f, d_in))
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        
        # Interface properties
        self.activation_dim = d_in
        self.dict_size = bottleneck_k
        
        # Config integration
        if hook_name is None:
            hook_name = f"blocks.{hook_layer}.hook_resid_post"
            
        self.cfg = CustomSAEConfig(
            model_name=model_name,
            d_in=d_in,
            d_sae=bottleneck_k,
            hook_name=hook_name,
            hook_layer=hook_layer,
            architecture="compsae",
            activation_fn_str="relu",
            apply_b_dec_to_input=True,
        )

    def encode(self, input_acts):
        # Feature extraction stream
        f_acts = torch.relu((input_acts - self.b_dec) @ self.W_enc_f + self.b_enc_f)
        
        # Composition stream with attention
        c_acts = torch.relu(f_acts @ self.W_enc_c + self.b_enc_c)
        c_acts = c_acts.unsqueeze(1)  # Add sequence dimension for attention
        
        # Multi-head attention for feature composition
        attn_out, _ = self.mha(c_acts, c_acts, c_acts)
        attn_out = attn_out.squeeze(1)
        
        # Bottleneck layer with sparsity
        bottleneck = torch.relu(attn_out @ self.bottleneck)
        
        # Apply top-k sparsity
        values, indices = torch.topk(bottleneck, k=self.dict_size, dim=-1)
        sparse_bottleneck = torch.zeros_like(bottleneck)
        sparse_bottleneck.scatter_(-1, indices, values)
        
        return sparse_bottleneck, f_acts

    def decode(self, bottleneck, f_acts):
        # Main decoding
        main_out = bottleneck @ self.W_dec
        
        # Residual connection from feature extractor
        res_out = f_acts @ self.W_res
        
        # Combine outputs
        return main_out + res_out + self.b_dec

    def forward(self, acts, output_features=False):
        bottleneck, f_acts = self.encode(acts)
        decoded = self.decode(bottleneck, f_acts)
        if output_features:
            return decoded, bottleneck
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


class VanillaTrainer(SAETrainer):
    """Trainer for Vanilla Sparse Autoencoder using L1 regularization."""
    def __init__(self,
                 activation_dim=512,
                 dict_size=32,  # Bottleneck size
                 d_f=256,  # Feature extractor dim
                 d_c=128,  # Composition network dim
                 n_heads=8,
                 lr=3e-4,  # As specified
                 l1_penalty=1e-1,
                 warmup_steps=1000,
                 resample_steps=None,
                 seed=None,
                 device=None,
                 layer=None,
                 lm_name=None,
                 wandb_name='CompSAETrainer',
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

        # Initialize compositional autoencoder
        self.ae = CompSAE(
            d_in=activation_dim,
            d_f=d_f,
            d_c=d_c,
            n_heads=n_heads,
            bottleneck_k=dict_size
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
        x_hat, f = self.ae(x, output_features=True)
        l2_loss = torch.linalg.norm(x - x_hat, dim=-1).mean()
        l1_loss = f.norm(p=1, dim=-1).mean()

        if self.steps_since_active is not None:
            # Update steps_since_active
            deads = (f == 0).all(dim=0)
            self.steps_since_active[deads] += 1
            self.steps_since_active[~deads] = 0
        
        loss = l2_loss + self.l1_penalty * l1_loss

        if not logging:
            return loss
        else:
            return namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])(
                x, x_hat, f,
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
    model_name: str = "facebook/opt-125m",
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
        torch_dtype=torch.float32,  # OPT models work with float32
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Get activation dimension from model config
    activation_dim = model.config.hidden_size
    
    # Setup dataset and dataloader
    dataset = load_dataset("monology/pile-uncopyrighted", streaming=True)
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=context_length,
            return_tensors="pt",
            padding="max_length"
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    dataloader = DataLoader(
        tokenized_dataset["train"],
        batch_size=sae_batch_size,
        shuffle=True
    )
    
    # Get layer module for activation extraction
    submodule = model.model.layers[layer]
    submodule_name = f"resid_post_layer_{layer}"

    # Initialize trainer  
    trainer = VanillaTrainer(
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
    dataloader_iter = iter(dataloader)
    for step in range(steps):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
            
        # Get activations from the model
        with torch.no_grad():
            inputs = batch["input_ids"].to(device)
            outputs = model(inputs, output_hidden_states=True)
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
from typing import Any, Optional, List, Dict, Union, Tuple
from tqdm import tqdm

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
        dict_size=32,  # Bottleneck size k=32 as specified
        num_tokens=1_000_000,  # 1M tokens for training
        out_dir=run_dir,
        device="cuda" if torch.cuda.is_available() else "cpu",
        learning_rate=3e-4,  # As specified
        sae_batch_size=512,  # As specified
    )

def run(out_dir: str):
    """Run the SAE training experiment and save results."""
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
        dict_size=32,  # Bottleneck size k=32 as specified
        num_tokens=1_000_000,  # 1M tokens for training
        out_dir=run_dir,
        device="cuda" if torch.cuda.is_available() else "cpu",
        learning_rate=3e-4,  # As specified
        sae_batch_size=512,  # As specified
    )

if __name__ == "__main__":
    run("results/sae_variants")
