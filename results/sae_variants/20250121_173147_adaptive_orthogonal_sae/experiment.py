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
class OrthogonalSAE(nn.Module):
    """Implementation of Sparse Autoencoder with k-winners-take-all activation."""
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        hook_layer: int,
        model_name: str = "EleutherAI/pythia-70m-deduped",
        hook_name: Optional[str] = None,
        alpha: float = 0.01,  # Base orthogonality loss weight
        initial_features: float = 0.25,
        feature_growth_rate: float = 0.1,
        feature_growth_steps: int = 1000,
        ortho_curriculum_steps: int = 5000,  # Steps for orthogonality curriculum
        max_alpha: float = 0.1,  # Maximum orthogonality weight
    ):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.initial_features = int(d_sae * initial_features)
        self.current_features = self.initial_features
        self.feature_growth_rate = feature_growth_rate
        self.feature_growth_steps = feature_growth_steps
        
        # Initialize feature mask
        self.feature_mask = torch.zeros(d_sae, dtype=torch.bool)
        self.feature_mask[:self.initial_features] = True
        
        # Initialize weights and biases with improved initialization
        self.W_enc = nn.Parameter(torch.nn.init.orthogonal_(torch.empty(d_in, d_sae)))
        self.W_dec = nn.Parameter(torch.nn.init.orthogonal_(torch.empty(d_sae, d_in)))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        
        # Add layer normalization and feature-wise normalization
        self.layer_norm = nn.LayerNorm(d_sae)
        self.feature_norm = nn.LayerNorm(d_sae, elementwise_affine=False)
        
        # K-winners-take-all parameters
        self.k = int(d_sae * 0.1)  # Start with 10% active features
        self.temperature = 0.1  # Temperature for soft k-winners
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self.normalize_weights = True  # Add weight normalization flag
        
        # Add numerical stability epsilon
        self.eps = 1e-8
        
        # Add properties to match the interface expected by CustomTrainer
        self.activation_dim = d_in
        self.dict_size = d_sae

        # Add CustomSAEConfig integration
        if hook_name is None:
            hook_name = f"blocks.{hook_layer}.hook_resid_post"

        # Initialize orthogonality parameters
        self.alpha = alpha  # Current alpha value
        self.base_alpha = alpha  # Starting alpha value
        self.max_alpha = max_alpha  # Maximum alpha value
        self.ortho_curriculum_steps = ortho_curriculum_steps  # Steps for curriculum
        
        self.cfg = CustomSAEConfig(
            model_name=model_name,
            d_in=d_in,
            d_sae=d_sae,
            hook_name=hook_name,
            hook_layer=hook_layer,
            architecture="Custom",
            activation_fn_str="relu",
            apply_b_dec_to_input=True,
        )

    def k_winners_take_all(self, x):
        """Soft k-winners-take-all activation function."""
        # Handle arbitrary batch dimensions
        orig_shape = x.shape
        x_2d = x.view(-1, x.shape[-1])  # Flatten all batch dims
        
        # Sort activations along feature dimension
        values, indices = torch.sort(x_2d, dim=-1, descending=True)
        
        # Create k-hot mask
        k_values = values[:, :self.k]
        k_threshold = k_values[:, -1].unsqueeze(-1)
        
        # Compute soft mask using temperature
        soft_mask = torch.sigmoid((x_2d - k_threshold) / self.temperature)
        
        # Apply feature mask for progressive activation
        if hasattr(self, 'feature_mask'):
            soft_mask = soft_mask * self.feature_mask.to(x.device)
        
        # Apply mask and restore original shape
        result = (x_2d * soft_mask).view(orig_shape)
        return result

    def encode(self, input_acts):
        pre_acts = (input_acts - self.b_dec) @ self.W_enc + self.b_enc
        # Apply layer and feature normalization
        normalized = self.feature_norm(self.layer_norm(pre_acts))
        # Apply k-winners-take-all activation
        acts = self.k_winners_take_all(normalized)
        return acts

    def decode(self, acts):
        return (acts @ self.W_dec) + self.b_dec

    def forward(self, acts, output_features=False):
        # Normalize weights
        if self.normalize_weights:
            with torch.no_grad():
                self.W_enc.data = self.W_enc.data / (self.W_enc.data.norm(dim=0, keepdim=True) + self.eps)
                self.W_dec.data = self.W_dec.data / (self.W_dec.data.norm(dim=1, keepdim=True) + self.eps)
        
        # Forward pass with k-winners-take-all
        pre_acts = acts @ self.W_enc + self.b_enc
        normalized = self.feature_norm(self.layer_norm(pre_acts))
        encoded = self.k_winners_take_all(normalized)
        decoded = encoded @ self.W_dec + self.b_dec
        
        if output_features:
            return decoded, encoded
        return decoded
    
    def grow_features(self, step):
        """Gradually increase the number of active features."""
        if step % self.feature_growth_steps == 0 and self.current_features < self.d_sae:
            new_features = min(
                int(self.current_features * (1 + self.feature_growth_rate)),
                self.d_sae
            )
            self.feature_mask = self.feature_mask.to(self.W_enc.device)
            self.feature_mask[self.current_features:new_features] = True
            self.current_features = new_features

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
class OrthogonalTrainer(SAETrainer):
    """Trainer for Sparse Autoencoder with orthogonality constraints."""
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
        # Enable gradient computation for all parameters
        torch.set_grad_enabled(True)
        
        # Initialize weight normalization parameters
        self.weight_norm_eps = 1e-8
        self.normalize_weights = True
        super().__init__(seed)

        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Initialize autoencoder
        self.ae = OrthogonalSAE(d_in=activation_dim, d_sae=dict_size, hook_layer=layer, model_name=lm_name)

        # Initialize training parameters
        self.lr = lr
        self.alpha = alpha  # Current orthogonality weight
        self.base_alpha = alpha  # Initial orthogonality weight
        self.max_alpha = 0.1  # Maximum orthogonality weight
        self.ortho_curriculum_steps = 5000  # Steps for orthogonality curriculum
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
        # Initialize per-feature learning rates
        self.feature_lr = torch.ones(self.ae.dict_size, device=self.device) * lr
        self.feature_lr_min = lr * 0.1
        self.feature_lr_max = lr * 10.0
        
        # Initialize optimizer with constrained parameters
        constrained_params = [self.ae.W_dec]  # Only decoder weights are constrained
        param_groups = [
            {'params': [self.ae.W_enc], 'lr': lr},
            {'params': constrained_params, 'lr': lr},
            {'params': [self.ae.b_enc, self.ae.b_dec], 'lr': lr}
        ]
        self.optimizer = ConstrainedAdam(param_groups, constrained_params, lr)
        
        # Gradient accumulation settings
        self.grad_accum_steps = 4
        self.current_accum_step = 0
        
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
    
    def loss(self, x, step=0, logging=False, **kwargs):
        # Implement loss warm-up
        warmup_factor = min(step / (self.warmup_steps / 2), 1.0)
        
        # Forward pass through autoencoder
        x_hat, f = self.ae(x, output_features=True)
        
        # Reconstruction loss with gradient clipping
        x_diff = torch.clamp(x_hat - x.detach(), -100, 100)  # Detach input
        l2_loss = (x_diff ** 2).mean()
        
        # L1 sparsity loss on activations with stability
        l1_loss = torch.clamp(f.abs().mean(), 0, 100)
        
        # Improved orthogonality loss calculation
        W_enc = self.ae.W_enc  # No need for requires_grad_() as Parameter already has it
        # Normalize with epsilon for stability
        W_enc_norm = W_enc / (W_enc.norm(dim=0, keepdim=True) + self.ae.eps)
        
        # Calculate gram matrix with stable computation
        gram = torch.mm(W_enc_norm.t(), W_enc_norm)
        
        # Calculate orthogonality loss with curriculum
        mask = ~torch.eye(gram.shape[0], dtype=torch.bool, device=gram.device)
        current_alpha = min(self.base_alpha + (self.max_alpha - self.base_alpha) * (step / self.ortho_curriculum_steps), self.max_alpha)
        self.alpha = current_alpha  # Update current alpha for logging
        ortho_loss = torch.clamp((gram[mask] ** 2).mean(), 0, 10)

        if self.steps_since_active is not None:
            deads = (f.abs().mean(dim=0) < self.ae.eps)
            self.steps_since_active[deads] += 1
            self.steps_since_active[~deads] = 0
        
        # Apply warm-up to all constraint losses
        loss = l2_loss + warmup_factor * (self.l1_penalty * l1_loss + self.ae.alpha * ortho_loss)
        
        # Check for NaN and replace with large but finite value
        if torch.isnan(loss):
            loss = torch.tensor(1e6, device=loss.device, dtype=loss.dtype)

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
        # Ensure activations require gradients
        activations = activations.to(self.device).requires_grad_()

        # Grow features if needed
        self.ae.grow_features(step)

        # Forward pass with gradient tracking
        loss = self.loss(activations, step=step, logging=False)
        
        # Scale loss for gradient accumulation
        loss = loss / self.grad_accum_steps
        
        # Check if loss requires gradients
        if not loss.requires_grad:
            print("Warning: Loss does not require gradients")
            return
            
        loss.backward()
        
        # Update on final accumulation step
        self.current_accum_step += 1
        if self.current_accum_step >= self.grad_accum_steps:
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.ae.parameters(), max_norm=1.0)
            
            # Update per-feature learning rates based on gradient magnitudes
            with torch.no_grad():
                grad_norms = self.ae.W_enc.grad.norm(dim=0)
                lr_scale = torch.clamp(1.0 / (grad_norms + self.eps), 0.1, 10.0)
                self.feature_lr *= lr_scale
                self.feature_lr.clamp_(self.feature_lr_min, self.feature_lr_max)
                
                # Apply feature-wise learning rates
                self.ae.W_enc.grad *= self.feature_lr.unsqueeze(0)
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
            self.current_accum_step = 0

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

    # Setup dataset and buffer with absolute minimal batch sizes
    generator = hf_dataset_to_generator("monology/pile-uncopyrighted")
    max_retries = 5
    current_try = 0
    
    while current_try < max_retries:
        try:
            # Start with absolute minimum batch sizes
            n_ctxs = 1  # Single context window
            ctx_len = 1  # Single token context
            refresh_size = 1  # Single refresh batch
            out_size = 1  # Single output batch
            
            print(f"Attempt {current_try + 1}: Using absolute minimal configuration - "
                  f"n_ctxs={n_ctxs}, ctx_len={ctx_len}, "
                  f"refresh_size={refresh_size}, out_size={out_size}")
            
            # Clear CUDA cache and garbage collect before each attempt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            import gc
            gc.collect()
            
            try:
                activation_buffer = ActivationBuffer(
                    generator,
                    model,
                    submodule,
                    n_ctxs=n_ctxs,
                    ctx_len=ctx_len,
                    refresh_batch_size=refresh_size,
                    out_batch_size=out_size,
                    io="out",
                    d_submodule=activation_dim,
                    device=device
                )
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    import gc
                    gc.collect()
                raise  # Re-raise the exception after cleanup
            print("Successfully initialized buffer")
            break
        except Exception as e:
            current_try += 1
            print(f"Attempt {current_try}: Error initializing buffer: {str(e)}")
            if current_try == max_retries:
                print("Failed to initialize buffer after maximum retries")
                print("Final error:", str(e))
                raise
            print("Retrying with smaller batch sizes...")
            continue

    # Initialize trainer  
    trainer = OrthogonalTrainer(
        activation_dim=activation_dim,
        dict_size=dict_size,
        lr=learning_rate * 0.1,  # Reduce learning rate for stability
        l1_penalty=sparsity_penalty * 0.5,  # Reduce sparsity penalty
        warmup_steps=warmup_steps * 2,  # Double warmup period
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
        # Get loss info using loss() method with logging=True
        loss_info = trainer.loss(activations, logging=True)
        trainer.update(step, activations)
        
        # Store loss metrics
        if hasattr(loss_info, 'losses'):
            training_log.append(loss_info.losses)
            
            if step % 100 == 0:
                print(f"Step {step}: {loss_info.losses}")
                
                if wandb_logging and wandb_entity and wandb_project:
                    import wandb
                    wandb.log(loss_info.losses, step=step)

    # Prepare final results
    final_info = {
        "training_steps": steps,
        "final_loss": training_log[-1]['loss'] if training_log else 0.0,
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

# Make imports relative to root directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Evaluation imports
import evals.absorption.main as absorption
import evals.autointerp.main as autointerp
import evals.core.main as core
import evals.scr_and_tpp.main as scr_and_tpp
import evals.sparse_probing.main as sparse_probing
import evals.unlearning.main as unlearning

RANDOM_SEED = 42




MODEL_CONFIGS = {
    # "EleutherAI/pythia-70m-deduped": {"batch_size": 512, "dtype": "float32", "layers": [3, 4], "d_model": 512},
    "google/gemma-2-2b": {"batch_size": 32, "dtype": "bfloat16", "layers": [5, 12, 19], "d_model": 2304},
}



def evaluate_trained_sae(
    selected_saes: list[tuple[str, Any]],
    model_name: str,
    eval_types: list[str],
    device: str,
    llm_batch_size: Optional[int] = None,
    llm_dtype: Optional[str] = None,
    api_key: Optional[str] = None,
    force_rerun: bool = False,
    save_activations: bool = False,
    out_dir: str = "eval_results",
):
    """Run evaluations for the given model and SAE.
    
    Args:
        sae_model: The trained SAE model to evaluate
        model_name: Name of the base LLM model
        eval_types: List of evaluation types to run
        device: Device to run evaluations on
        llm_batch_size: Batch size for LLM inference
        llm_dtype: Data type for LLM ('float32' or 'bfloat16')
        api_key: Optional API key for certain evaluations
        force_rerun: Whether to force rerun of evaluations
        save_activations: Whether to save activations during evaluation
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model: {model_name}")
    
    if llm_batch_size is None or llm_dtype is None:
        config = MODEL_CONFIGS[model_name]
        llm_batch_size = llm_batch_size or config["batch_size"]
        llm_dtype = llm_dtype or config["dtype"]
    
    selected_saes = selected_saes
    
    # Mapping of eval types to their functions
    # Try to load API key for autointerp if needed
    if "autointerp" in eval_types and api_key is None:
        try:
            api_key_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "openai_api_key.txt")
            with open(api_key_path) as f:
                api_key = f.read().strip()
        except FileNotFoundError:
            print("Warning: openai_api_key.txt not found. Autointerp evaluation will be skipped.")
    
    eval_runners = {
        "absorption": (
            lambda: absorption.run_eval(
                absorption.AbsorptionEvalConfig(
                    model_name=model_name,
                    random_seed=RANDOM_SEED,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                out_dir,
                force_rerun,
            )
        ),
        "autointerp": (
            lambda: autointerp.run_eval(
                autointerp.AutoInterpEvalConfig(
                    model_name=model_name,
                    random_seed=RANDOM_SEED,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                api_key,
                out_dir,
                force_rerun,
            )
        ),
        "core": (
            lambda: core.multiple_evals(
                selected_saes=selected_saes,
                n_eval_reconstruction_batches=200,
                n_eval_sparsity_variance_batches=2000,
                eval_batch_size_prompts=16,
                compute_featurewise_density_statistics=False,
                compute_featurewise_weight_based_metrics=False,
                exclude_special_tokens_from_reconstruction=True,
                dataset="Skylion007/openwebtext",
                context_size=128,
                output_folder=out_dir,
                verbose=True,
                dtype=llm_dtype,
            )
        ),
        "scr": (
            lambda: scr_and_tpp.run_eval(
                scr_and_tpp.ScrAndTppEvalConfig(
                    model_name=model_name,
                    random_seed=RANDOM_SEED,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                out_dir,
                force_rerun,
            )
        ),
        "tpp": (
            lambda: scr_and_tpp.run_eval(
                scr_and_tpp.ScrAndTppEvalConfig(
                    model_name=model_name,
                    random_seed=RANDOM_SEED,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                out_dir,
                force_rerun,
            )
        ),
        "sparse_probing": (
            lambda: sparse_probing.run_eval(
                sparse_probing.SparseProbingEvalConfig(
                    model_name=model_name,
                    random_seed=RANDOM_SEED,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                out_dir,
                force_rerun,
            )
        ),
        # note that we automatically evaluate on the instruction tuned version of the model here
        "unlearning": (
            lambda: unlearning.run_eval(
                unlearning.UnlearningEvalConfig(
                    model_name=model_name+"-it",
                    random_seed=RANDOM_SEED,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                out_dir,
                force_rerun,
            )
        ),
    }
    
    
    # Run selected evaluations
    for eval_type in eval_types:
        if eval_type in eval_runners:
            print(f"\nRunning {eval_type} evaluation...")
            eval_runners[eval_type]()
        else:
            print(f"Warning: Unknown evaluation type {eval_type}")
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
    num_tokens = 100000 # Increase training tokens for better convergence
    device = "cuda" if torch.cuda.is_available() else "cpu"
    no_wandb_logging = False # Set default wandb logging flag
    
    saes = []
    alpha = 0.1  # Orthogonality loss weight for Run 1
    for layer in layers:
        saes.append(run_sae_training(
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
            learning_rate=1e-4,  # Reduced learning rate
            sparsity_penalty=0.04,
            warmup_steps=1000,
            seed=42,
            wandb_logging=not no_wandb_logging,
            wandb_entity=None,
            wandb_project=None
            ))        



    device = "cuda" if torch.cuda.is_available() else "cpu"


    # Note: Unlearning is not recommended for models with < 2B parameters and we recommend an instruct tuned model
    # Unlearning will also require requesting permission for the WMDP dataset (see unlearning/README.md)
    # Absorption not recommended for models < 2B parameters

    # Select your eval types here.
    eval_types = [
        # "absorption",
        # "autointerp",
        "core",
        # "scr",
        # "tpp",
        # "sparse_probing",
        "unlearning",
    ]

    if "autointerp" in eval_types:
        try:
            with open("openai_api_key.txt") as f:
                api_key = f.read().strip()
        except FileNotFoundError:
            raise Exception("Please create openai_api_key.txt with your API key")
    else:
        api_key = None

    save_activations = False

    for k in range(len(layers)):
        selected_saes = [(f"{model_name}_layer_{layers[k]}_sae", saes[k])]
        for sae_name, sae in selected_saes:
            sae = sae.to(dtype=str_to_dtype(llm_dtype))
            sae.cfg.dtype = llm_dtype

        evaluate_trained_sae(
            selected_saes=selected_saes,
            model_name=model_name,
            eval_types=eval_types,
            device=device,
            llm_batch_size=llm_batch_size,
            llm_dtype=llm_dtype,
            api_key=api_key,
            force_rerun=False,
            save_activations=False,
            out_dir=save_dir
        )
