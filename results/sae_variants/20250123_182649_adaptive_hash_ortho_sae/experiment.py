import torch
import os
import torch.nn as nn
import numpy as np
try:
    import faiss
except ImportError:
    print("Faiss not found. Installing faiss-cpu...")
    import subprocess
    subprocess.check_call(["pip", "install", "faiss-cpu"])
    import faiss
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
class AdaptiveHashSAE(nn.Module):
    """Implementation of Adaptive Resolution Hash Orthogonal SAE with Momentum Feature Prioritization and Residual Learning."""
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        hook_layer: int,
        model_name: str = "EleutherAI/pythia-70m-deduped",
        hook_name: Optional[str] = None,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        # Initialize weights with Kaiming initialization
        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
        nn.init.kaiming_uniform_(self.W_enc, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.W_dec, nonlinearity='relu')
        
        # Initialize biases to small positive values for ReLU
        self.b_enc = nn.Parameter(torch.ones(d_sae) * 0.01)
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        
        # Gradient clipping threshold
        self.grad_clip = 1.0
        
        # Residual learning parameters with improved initialization
        self.residual_scale = residual_scale
        self.skip_gate = nn.Parameter(torch.ones(d_in) * residual_scale)  # Per-dimension skip gates
        self.skip_gate_norm = nn.LayerNorm(d_in)  # Normalize skip connection
        
        # EMA tracking of feature importance
        self.register_buffer('feature_ema', torch.zeros(d_sae))
        self.ema_decay = 0.99
        
        # Training step counter
        self.register_buffer('training_steps', torch.zeros(1, dtype=torch.long))
        
        # LSH parameters with optimized update schedule
        self.n_bits = 4  # Simplified fixed bit depth
        self.lsh_index = None
        self.lsh_update_frequency = 1000  # Reduced frequency to prevent overhead
        self.exact_check_interval = 2000  # Increased to reduce computation
        self.max_batch_size = 4096  # Limit batch size for LSH updates
        self.initialize_lsh()
        
        # Add properties to match the interface expected by CustomTrainer
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
            architecture="Custom",
            activation_fn_str="relu",
            apply_b_dec_to_input=True,
        )

    def initialize_lsh(self):
        """Initialize LSH index with exact backup"""
        d = self.W_enc.shape[0]
        # Main LSH index
        self.lsh_index = faiss.IndexLSH(d, self.n_bits)
        # Exact index for periodic validation
        self.exact_index = faiss.IndexFlatL2(d)
        self.last_exact_check = 0
        self.exact_check_interval = 1000  # Check every 1000 steps
        self.max_drift_threshold = 0.1  # Maximum allowed drift before recomputing LSH
    
    def update_feature_importance(self, acts):
        """Update EMA of feature activations"""
        with torch.no_grad():
            current_importance = (acts > 0).float().mean(0)  # Average over batch dimension
            self.feature_ema = self.feature_ema.mul(self.ema_decay).add(
                current_importance * (1 - self.ema_decay)
            )
    
    def encode(self, input_acts):
        try:
            # Ensure input is on correct device and has proper dtype
            input_acts = input_acts.to(device=self.device, dtype=self.dtype)
            
            # Apply importance-weighted encoding with adaptive learning rates
            importance_weights = self.feature_ema.view(-1).to(device=self.device, dtype=self.dtype)
            # Reduced importance scaling with minimum threshold
            importance_scale = torch.clamp(torch.sqrt(importance_weights + 1e-6), min=0.3)
            
            # More balanced learning rate scaling
            lr_scale = torch.where(
                importance_weights > 0.05,  # Lowered threshold
                1.0 / torch.sqrt(importance_weights + 1e-6),
                torch.ones_like(importance_weights, device=self.device, dtype=self.dtype) * 0.3  # Increased minimum
            )
            
            # Ensure weights are on correct device and dtype
            weighted_W_enc = self.W_enc.to(device=self.device, dtype=self.dtype)
            for i in range(self.W_enc.shape[1]):
                weighted_W_enc[:, i] *= importance_scale[i] * lr_scale[i]
                
            # Compute activations with minimum threshold
            if not self.training:
                pre_acts = (input_acts - self.b_dec.to(device=self.device, dtype=self.dtype)) @ weighted_W_enc
                pre_acts = pre_acts + self.b_enc.to(device=self.device, dtype=self.dtype)
                acts = torch.relu(pre_acts)
                # Apply minimum activation threshold
                acts = torch.where(acts > 0.1, acts, torch.zeros_like(acts))
                return acts.to(device=self.device, dtype=self.dtype)
                
            # Update LSH during training
            if self.training_steps.item() % self.lsh_update_frequency == 0:
                try:
                    with torch.no_grad():
                        self.validate_lsh_accuracy(input_acts.detach())
                except Exception as e:
                    print(f"LSH validation error: {e}")
            
            try:
                # Convert to numpy safely for LSH
                with torch.no_grad():
                    input_np = input_acts.detach().cpu().numpy().astype(np.float32)
                    if len(input_np.shape) > 2:
                        input_np = input_np.reshape(-1, input_np.shape[-1])
                    _, I = self.lsh_index.search(input_np, 1)
            except Exception as e:
                print(f"LSH search error: {e}")
                I = None
            
            # Compute activations
            pre_acts = (input_acts - self.b_dec.to(device=self.device, dtype=self.dtype)) @ weighted_W_enc
            pre_acts = pre_acts + self.b_enc.to(device=self.device, dtype=self.dtype)
            acts = torch.relu(pre_acts)
            
            # Update feature importance tracking
            if self.training:
                with torch.no_grad():
                    self.update_feature_importance(acts.detach())
                
            return acts.to(device=self.device, dtype=self.dtype)
            
        except Exception as e:
            print(f"Encode error: {e}")
            # Return zero activations on critical error
            return torch.zeros(input_acts.shape[:-1] + (self.W_enc.shape[1],), 
                             device=self.device, dtype=self.dtype)
        
    def validate_lsh_accuracy(self, input_acts):
        """Check LSH accuracy against exact search with improved batching and timeouts"""
        try:
            with torch.no_grad():
                # Convert input to numpy safely
                try:
                    input_float32 = input_acts.to(torch.float32).cpu()
                    input_np = input_float32.numpy()
                except Exception as e:
                    print(f"Input conversion error: {e}")
                    return

                # Early validation and cleanup
                if not np.all(np.isfinite(input_np)):
                    input_np = np.nan_to_num(input_np, nan=0.0, posinf=0.0, neginf=0.0)
                
                if len(input_np.shape) > 2:
                    input_np = input_np.reshape(-1, input_np.shape[-1])
                
                if input_np.shape[1] != self.W_enc.shape[0]:
                    print(f"Dimension mismatch: input {input_np.shape[1]}, expected {self.W_enc.shape[0]}")
                    return

                # Use very small batches with strict timeouts
                batch_size = min(256, input_np.shape[0])  # Even smaller batches
                accuracy_sum = 0
                n_batches = 0
                max_time_per_batch = 1  # Stricter timeout
                
                import time
                from contextlib import contextmanager

                @contextmanager
                def timeout_context(seconds):
                    start = time.time()
                    yield
                    if time.time() - start > seconds:
                        raise TimeoutError(f"Processing timed out after {seconds} seconds")

                # Process batches with stricter controls
                for i in range(0, min(input_np.shape[0], 5000), batch_size):  # Limit total samples
                    try:
                        with timeout_context(max_time_per_batch):
                            batch = input_np[i:i + batch_size]
                            if not np.all(np.isfinite(batch)):
                                continue
                                
                            if batch.shape[0] == 0 or batch.shape[1] != self.W_enc.shape[0]:
                                continue
                                
                            # Simplified neighbor search
                            _, I_lsh = self.lsh_index.search(batch, 1)
                            _, I_exact = self.exact_index.search(batch, 1)
                            
                            batch_accuracy = (I_lsh == I_exact).mean()
                            if np.isfinite(batch_accuracy):
                                accuracy_sum += batch_accuracy
                                n_batches += 1
                                
                            # Early exit if we have enough samples
                            if n_batches >= 10:
                                break
                                
                    except Exception as e:
                        print(f"Batch {i} error: {str(e)[:100]}")
                        continue

                if n_batches == 0:
                    return
                    
                accuracy = accuracy_sum / n_batches
                
                # Very conservative rebuild policy
                if accuracy < 0.3:
                    try:
                        with timeout_context(5):  # Shorter rebuild timeout
                            self.initialize_lsh()
                            max_rebuild_size = min(5000, input_np.shape[0])  # Even smaller rebuild
                            rebuild_data = input_np[:max_rebuild_size]
                            if np.all(np.isfinite(rebuild_data)):
                                self.lsh_index.add(rebuild_data)
                                self.exact_index.add(rebuild_data)
                    except Exception as e:
                        print(f"LSH rebuild failed: {str(e)[:100]}")
                        
        except Exception as e:
            print(f"LSH validation failed: {str(e)[:100]}")

    def decode(self, acts):
        # Combine learned features with skip connection
        learned_features = (acts @ self.W_dec) + self.b_dec
        return learned_features

    def forward(self, acts, output_features=False):
        encoded = self.encode(acts)
        decoded = self.decode(encoded)
        
        # Normalize and apply per-dimension skip connections
        skip_contribution = self.skip_gate_norm(acts)
        skip_contribution = skip_contribution * self.skip_gate
        
        # Combine learned features with skip connection
        output = decoded + skip_contribution
        
        # Track skip connection usage for monitoring
        with torch.no_grad():
            self.skip_usage = (skip_contribution.abs() > 0.1).float().mean().item()
            
        if output_features:
            return output, encoded
        return output

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device = kwargs.get("device", None)
        dtype = kwargs.get("dtype", None)
        if device:
            self.device = device
        if dtype:
            self.dtype = dtype
        return self


class ConstrainedAdaptiveAdam(torch.optim.Adam):
    """A variant of Adam with per-feature learning rates and unit norm constraints."""
    def __init__(self, params, constrained_params, lr, feature_ema):
        super().__init__(params, lr=lr)
        self.constrained_params = list(constrained_params)
        self.feature_ema = feature_ema
        
    def step(self, closure=None):
        with torch.no_grad():
            # Compute adaptive learning rates
            importance_weights = self.feature_ema.view(-1)
            lr_scale = torch.where(
                importance_weights > 0.1,
                1.0 / torch.sqrt(importance_weights + 1e-6),
                torch.ones_like(importance_weights) * 0.1
            )
            
            for p in self.constrained_params:
                normed_p = p / p.norm(dim=0, keepdim=True)
                # Scale gradients by feature importance
                p.grad *= lr_scale.view(1, -1)
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
class AdaptiveHashTrainer(SAETrainer):
    """Trainer for Adaptive Hash SAE with feature importance tracking."""
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
                 grad_clip=1.0,  # Add gradient clipping
                 min_activation_rate=0.01,  # Minimum desired activation rate
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
        self.ae = AdaptiveHashSAE(d_in=activation_dim, d_sae=dict_size, hook_layer=layer, model_name=lm_name)

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

        # Initialize optimizer with constrained decoder weights and adaptive learning rates
        self.optimizer = ConstrainedAdaptiveAdam(
            self.ae.parameters(),
            [self.ae.W_dec],  # Constrain decoder weights
            lr=lr,
            feature_ema=self.ae.feature_ema
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
        
        # Increment training steps
        self.ae.training_steps += 1

        self.optimizer.zero_grad()
        loss = self.loss(activations)
        loss.backward()
        
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(self.ae.parameters(), self.grad_clip)
        
        self.optimizer.step()
        self.scheduler.step()

        # Monitor and log feature statistics
        with torch.no_grad():
            _, features = self.ae(activations, output_features=True)
            activation_rate = (features > 0).float().mean(0)
            dead_features = (activation_rate < self.min_activation_rate).sum().item()
            if step % 100 == 0:
                print(f"Step {step}: Dead features: {dead_features}")

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
    trainer = AdaptiveHashTrainer(
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
    num_tokens = 1000 # Set default number of tokens
    device = "cuda" if torch.cuda.is_available() else "cpu"
    no_wandb_logging = False # Set default wandb logging flag
    
    saes = []
    for layer in layers:
        # Initialize SAE with adaptive hashing
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
        "sparse_probing",
        # "unlearning",
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
