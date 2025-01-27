import torch
import os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from huggingface_hub import hf_hub_download
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from nnsight import LanguageModel
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
class TopKOrthogonalSAE(nn.Module):
    """Implementation of Sparse Autoencoder with Top-k Orthogonality Constraints."""
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        hook_layer: int,
        model_name: str = "EleutherAI/pythia-70m-deduped",
        hook_name: Optional[str] = None,
        topk_percentage: float = 0.001,  # 0.1%
        tau_init: float = 0.05,  # Reduced initial τ
        tau_momentum: float = 0.95,  # Increased momentum
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self.topk_percentage = topk_percentage
        self.tau = tau_init
        self.tau_momentum = tau_momentum
        self.moving_avg_corr = 0.0
        self.pair_counts = torch.zeros((d_sae, d_sae), device=self.device)
        self.pair_history = torch.zeros((d_sae, d_sae), device=self.device)
        self.history_decay = 0.99  # Decay factor for historical pair selection
        
        # Correlation pattern tracking
        self.corr_stats = {
            'mean': [],
            'std': [],
            'max': [],
            'median': []
        }
        self.corr_update_freq = 100  # Update correlation stats every 100 steps
        # Initialize weights properly
        nn.init.kaiming_uniform_(torch.empty(d_in, d_sae), mode='fan_in', nonlinearity='relu')
        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.W_enc, mode='fan_in', nonlinearity='relu')
        nn.init.orthogonal_(self.W_dec)
        
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

    def encode(self, input_acts):
        pre_acts = (input_acts - self.b_dec) @ self.W_enc + self.b_enc
        acts = torch.relu(pre_acts)
        return acts

    def decode(self, acts):
        return (acts @ self.W_dec) + self.b_dec

    def get_top_pairs(self, encoded, step):
        """Select top-k most correlated feature pairs and update adaptive τ."""
        B = encoded.size(0)
        # Compute pairwise correlations
        norm_encoded = encoded / (encoded.norm(dim=0, keepdim=True) + 1e-8)
        correlations = torch.abs(norm_encoded.T @ norm_encoded / B)
        
        # Get top k pairs excluding diagonal
        mask = torch.ones_like(correlations, dtype=torch.bool)
        mask.diagonal().fill_(False)
        correlations = correlations * mask
        
        k = int(self.topk_percentage * (self.dict_size * (self.dict_size-1) / 2))
        top_corr, indices = torch.topk(correlations.view(-1), k)
        row_idx = indices // self.dict_size
        col_idx = indices % self.dict_size
        
        # Update pair counts and history
        self.pair_counts[row_idx, col_idx] += 1
        self.pair_counts[col_idx, row_idx] += 1
        
        # Update pair history with exponential decay
        self.pair_history *= self.history_decay
        self.pair_history[row_idx, col_idx] += 1
        self.pair_history[col_idx, row_idx] += 1
        
        # Update moving average correlation and adapt τ
        curr_avg_corr = top_corr.mean().item()
        self.moving_avg_corr = (self.tau_momentum * self.moving_avg_corr + 
                              (1 - self.tau_momentum) * curr_avg_corr)
        # Implement gradual warmup for τ adaptation
        warmup_factor = min(1.0, step / 1000)  # Gradual warmup over 1000 steps
        target_tau = max(0.01, min(1.0, self.moving_avg_corr * 2))
        self.tau = warmup_factor * target_tau + (1 - warmup_factor) * self.tau_init
        
        return row_idx, col_idx, top_corr

    def orthogonality_loss(self, encoded, step):
        """Compute orthogonality loss for top pairs."""
        row_idx, col_idx, _ = self.get_top_pairs(encoded, step)
        if len(row_idx) == 0:
            return 0.0
            
        # Compute normalized dot products for selected pairs
        norm_encoded = encoded / (encoded.norm(dim=0, keepdim=True) + 1e-8)
        dot_products = torch.sum(
            norm_encoded[:, row_idx] * norm_encoded[:, col_idx], dim=0
        )
        return (dot_products ** 2).mean()

    def forward(self, acts, output_features=False):
        encoded = self.encode(acts)
        
        # Apply L2 normalization to decoder weights
        with torch.no_grad():
            self.W_dec.data = self.W_dec.data / (self.W_dec.data.norm(dim=0, keepdim=True) + 1e-8)
            
        decoded = self.decode(encoded)
        if output_features:
            return decoded, encoded
        return decoded

    def plot_correlation_patterns(self, save_dir):
        """Plot correlation statistics evolution."""
        if not self.corr_stats['mean']:
            return
        
        import matplotlib.pyplot as plt
        import os
        from scipy.cluster import hierarchy
        import numpy as np
        
        # Plot correlation evolution
        plt.figure(figsize=(12, 8))
        steps = list(range(0, len(self.corr_stats['mean']) * self.corr_update_freq, 
                          self.corr_update_freq))
        
        plt.plot(steps, self.corr_stats['mean'], label='Mean Correlation')
        plt.plot(steps, self.corr_stats['max'], label='Max Correlation')
        plt.plot(steps, self.corr_stats['median'], label='Median Correlation')
        plt.fill_between(steps, 
                        [m - s for m, s in zip(self.corr_stats['mean'], self.corr_stats['std'])],
                        alpha=0.2, label='±1 std')
        
        plt.xlabel('Training Steps')
        plt.ylabel('Correlation')
        plt.title('Feature Correlation Evolution')
        plt.legend()
        plt.grid(True)
        
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'correlation_evolution.png'))
        plt.close()
        
        # Generate feature clustering plot
        plt.figure(figsize=(15, 10))
        
        # Compute feature correlations
        with torch.no_grad():
            W_dec_norm = self.W_dec / (self.W_dec.norm(dim=0, keepdim=True) + 1e-8)
            feature_correlations = torch.abs(W_dec_norm.T @ W_dec_norm).cpu().numpy()
            
        # Perform hierarchical clustering
        linkage_matrix = hierarchy.linkage(feature_correlations, method='ward')
        
        # Plot dendrogram
        hierarchy.dendrogram(linkage_matrix, 
                           leaf_rotation=90,
                           leaf_font_size=8,
                           truncate_mode='lastp',
                           p=50)  # Show only last 50 merges
        
        plt.title('Feature Hierarchy Dendrogram')
        plt.xlabel('Feature Clusters')
        plt.ylabel('Distance')
        
        plt.savefig(os.path.join(save_dir, 'feature_clusters.png'))
        plt.close()

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
class TopKOrthogonalTrainer(SAETrainer):
    """Trainer for SAE with Top-k Orthogonality Constraints."""
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
                 wandb_name='TopKOrthogonalTrainer',
                 submodule_name=None,
                 topk_percentage=0.001,
                 tau=0.1,
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
        self.ae = TopKOrthogonalSAE(
            d_in=activation_dim, 
            d_sae=dict_size, 
            hook_layer=layer, 
            model_name=lm_name,
            topk_percentage=topk_percentage,
            tau_init=tau
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
    
    def loss(self, x, step=0, logging=False, **kwargs):
        x_hat, f = self.ae(x, output_features=True)
        l2_loss = torch.linalg.norm(x - x_hat, dim=-1).mean()
        l1_loss = f.norm(p=1, dim=-1).mean()
        ortho_loss = self.ae.orthogonality_loss(f, step)
        
        # Calculate pair stability metrics and correlation patterns
        if logging and step > 0:
            with torch.no_grad():
                stability = (self.ae.pair_history > 0.5).float().sum() / (self.ae.dict_size * self.ae.dict_size)
                
                # Update correlation statistics periodically
                if step % self.ae.corr_update_freq == 0:
                    norm_f = f / (f.norm(dim=0, keepdim=True) + 1e-8)
                    correlations = torch.abs(norm_f.T @ norm_f / f.size(0))
                    
                    # Mask out diagonal elements
                    mask = torch.ones_like(correlations, dtype=torch.bool)
                    mask.diagonal().fill_(False)
                    correlations = correlations[mask]
                    
                    self.ae.corr_stats['mean'].append(correlations.mean().item())
                    self.ae.corr_stats['std'].append(correlations.std().item())
                    self.ae.corr_stats['max'].append(correlations.max().item())
                    self.ae.corr_stats['median'].append(correlations.median().item())

        if self.steps_since_active is not None:
            deads = (f == 0).all(dim=0)
            self.steps_since_active[deads] += 1
            self.steps_since_active[~deads] = 0
        
        loss = l2_loss + self.l1_penalty * l1_loss + self.ae.tau * ortho_loss

        if not logging:
            return loss
        else:
            return namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])(
                x, x_hat, f,
                {
                    'l2_loss': l2_loss.item(),
                    'mse_loss': (x - x_hat).pow(2).sum(dim=-1).mean().item(),
                    'sparsity_loss': l1_loss.item(),
                    'loss': loss.item(),
                    'pair_stability': stability.item() if step > 0 else 0.0,
                    'corr_mean': self.ae.corr_stats['mean'][-1] if self.ae.corr_stats['mean'] else 0.0,
                    'corr_std': self.ae.corr_stats['std'][-1] if self.ae.corr_stats['std'] else 0.0,
                    'corr_max': self.ae.corr_stats['max'][-1] if self.ae.corr_stats['max'] else 0.0,
                    'corr_median': self.ae.corr_stats['median'][-1] if self.ae.corr_stats['median'] else 0.0
                }
            )

    def update(self, step, activations):
        activations = activations.to(self.device)

        self.optimizer.zero_grad()
        loss = self.loss(activations, step)
        loss.backward()
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
    trainer = TopKOrthogonalTrainer(
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
    
    # Train SAEs for each layer
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
        # Plot correlation patterns
        sae.plot_correlation_patterns(save_dir)
        print(f"Completed training SAE for layer {layer}")
