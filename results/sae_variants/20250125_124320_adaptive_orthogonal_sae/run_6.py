import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(sys.path)
import torch
import torch.nn as nn
import numpy as np
from collections import namedtuple
from huggingface_hub import hf_hub_download
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from nnsight import LanguageModel
from datasets import load_dataset
from torch.utils.data import IterableDataset

def hf_dataset_to_generator(dataset_name):
    """Load a HuggingFace dataset and convert it to a generator."""
    dataset = load_dataset(dataset_name, streaming=True)
    for item in dataset["train"]:
        if "text" in item:
            yield item["text"]
        elif "content" in item:
            yield item["content"]

class ActivationBuffer:
    """Buffer for storing and batching model activations."""
    def __init__(
        self,
        generator,
        model,
        submodule,
        n_ctxs,
        ctx_len,
        refresh_batch_size,
        out_batch_size,
        io="out",
        d_submodule=None,
        device="cuda",
    ):
        self.generator = generator
        self.model = model
        self.submodule = submodule
        self.n_ctxs = n_ctxs
        self.ctx_len = ctx_len
        self.refresh_batch_size = refresh_batch_size
        self.out_batch_size = out_batch_size
        self.io = io
        self.d_submodule = d_submodule
        self.device = device
        self.buffer = None
        self.buffer_idx = 0
        
    def __iter__(self):
        return self
        
    def __next__(self):
        if self.buffer is None or self.buffer_idx >= len(self.buffer):
            self._refresh_buffer()
            self.buffer_idx = 0
            
        batch = self.buffer[self.buffer_idx:self.buffer_idx + self.out_batch_size]
        self.buffer_idx += self.out_batch_size
        return batch.to(self.device)
    
    def _refresh_buffer(self):
        texts = []
        for _ in range(self.refresh_batch_size):
            try:
                texts.append(next(self.generator))
            except StopIteration:
                if not texts:
                    raise StopIteration
                break
                
        tokens = self.model.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.ctx_len,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**tokens)
            if self.io == "out":
                self.buffer = outputs.last_hidden_state
            else:
                self.buffer = outputs.hidden_states[self.submodule]
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
class CustomSAE(nn.Module):
    """Implementation of a Custom Sparse Autoencoder with orthogonality constraints."""
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        hook_layer: int,
        model_name: str = "EleutherAI/pythia-70m-deduped",
        hook_name: Optional[str] = None,
        tau: float = 0.1,
        topk_percentage: float = 0.001,  # 0.1%
        min_pairs: int = 10,  # Minimum number of pairs to consider
        correlation_threshold: float = 0.5,  # Initial correlation threshold
        prune_threshold: float = 0.8,  # Correlation threshold for pruning
        prune_window: int = 100,  # Steps between pruning checks
        importance_momentum: float = 0.99,  # Momentum for feature importance tracking
        sparsity_momentum: float = 0.99,  # Momentum for sparsity adaptation
        target_sparsity: float = 0.1  # Target activation sparsity
    ):
        super().__init__()
        self.W_enc = nn.Parameter(torch.zeros(d_in, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.steps_since_prune = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self.feature_importance = torch.ones(d_sae, device=self.device)  # Initialize importance scores
        self.sparsity_history = torch.zeros(d_sae, device=self.device)  # Track feature sparsity
        
        # Add properties to match the interface expected by CustomTrainer
        self.activation_dim = d_in
        self.dict_size = d_sae

        # Add CustomSAEConfig integration
        if hook_name is None:
            hook_name = f"blocks.{hook_layer}.hook_resid_post"

        self.tau_init = tau
        self.tau = tau
        self.tau_min = 0.01
        self.tau_max = 0.5
        self.tau_momentum = 0.9
        self.topk_percentage = topk_percentage
        self.min_pairs = min_pairs
        self.correlation_threshold = correlation_threshold
        self.correlation_momentum = 0.9  # For smooth threshold updates
        
        self.cfg = CustomSAEConfig(
            model_name=model_name,
            d_in=d_in,
            d_sae=d_sae,
            hook_name=hook_name,
            hook_layer=hook_layer,
            architecture="CustomOrtho",
            activation_fn_str="relu",
            apply_b_dec_to_input=True,
        )
        
        # Register L2 normalization hook
        self.register_forward_hook(self._l2_normalize_decoder)

    def encode(self, input_acts):
        pre_acts = (input_acts - self.b_dec) @ self.W_enc + self.b_enc
        acts = torch.relu(pre_acts)
        return acts

    def decode(self, acts):
        return (acts @ self.W_dec) + self.b_dec

    def _l2_normalize_decoder(self, module, input, output):
        with torch.no_grad():
            self.W_dec.data = self.W_dec.data / torch.norm(self.W_dec.data, dim=0, keepdim=True)
    
    def _get_top_pairs(self, features):
        # Compute pairwise dot products
        dot_products = torch.abs(torch.matmul(features.T, features))
        # Zero out diagonal
        dot_products.fill_diagonal_(0)
        
        # Get correlation distribution statistics
        mean_corr = dot_products.mean()
        std_corr = dot_products.std()
        
        # Update correlation threshold using momentum
        target_threshold = mean_corr + 2 * std_corr  # Target pairs > 2 standard deviations
        self.correlation_threshold = (
            self.correlation_momentum * self.correlation_threshold +
            (1 - self.correlation_momentum) * target_threshold.item()
        )
        
        # Select pairs above threshold
        mask = dot_products > self.correlation_threshold
        n_pairs = max(self.min_pairs, mask.sum() // 2)  # Ensure minimum pairs
        
        # Get top-k pairs
        values, indices = torch.topk(dot_products.view(-1), k=n_pairs)
        row_idx = indices // features.size(1)
        col_idx = indices % features.size(1)
        
        # Update tau based on correlation magnitudes
        with torch.no_grad():
            mean_correlation = values.mean().item()
            target_correlation = 0.1  # Target average correlation
            tau_update = self.tau_momentum * self.tau + (1 - self.tau_momentum) * (
                self.tau * (mean_correlation / target_correlation)
            )
            self.tau = torch.clamp(torch.tensor(tau_update), self.tau_min, self.tau_max).item()
        
        return row_idx, col_idx, values

    def forward(self, acts, output_features=False):
        encoded = self.encode(acts)
        decoded = self.decode(encoded)
        
        # Update sparsity history
        with torch.no_grad():
            current_sparsity = (encoded == 0).float().mean(dim=0)
            self.sparsity_history = (
                self.sparsity_momentum * self.sparsity_history +
                (1 - self.sparsity_momentum) * current_sparsity
            )
            
            # Adjust l1 penalty based on sparsity
            sparsity_error = self.sparsity_history.mean() - self.target_sparsity
            self.l1_penalty = torch.clamp(
                self.l1_penalty * (1 + 0.1 * sparsity_error),
                min=0.01,
                max=0.2
            )
        
        # Check if pruning is needed
        self.steps_since_prune += 1
        if self.steps_since_prune >= self.prune_window:
            self._prune_correlated_neurons(encoded)
            self.steps_since_prune = 0
        
        if output_features:
            return decoded, encoded
        return decoded
        
    def _prune_correlated_neurons(self, features):
        with torch.no_grad():
            # Update feature importance scores
            feature_activity = torch.abs(features).mean(dim=0)
            self.feature_importance = (
                self.importance_momentum * self.feature_importance + 
                (1 - self.importance_momentum) * feature_activity
            )
            
            # Compute correlation matrix
            features_norm = features - features.mean(dim=0, keepdim=True)
            features_norm = features_norm / (features_norm.std(dim=0, keepdim=True) + 1e-8)
            corr_matrix = torch.abs(torch.matmul(features_norm.T, features_norm)) / features.size(0)
            corr_matrix.fill_diagonal_(0)
            
            # Find highly correlated neurons
            max_corr, _ = corr_matrix.max(dim=1)
            
            # Adjust pruning threshold based on feature importance
            importance_percentile = (
                self.feature_importance.unsqueeze(1) > self.feature_importance
            ).float().mean(dim=1)
            adjusted_threshold = self.prune_threshold * (1 + importance_percentile)
            to_prune = max_corr > adjusted_threshold
            
            if to_prune.any():
                # Reinitialize pruned neurons
                n_pruned = to_prune.sum().item()
                print(f"Pruning {n_pruned} highly correlated neurons")
                
                # Sample random directions for reinitialization
                new_dirs = torch.randn(n_pruned, self.W_enc.size(0), device=self.device)
                new_dirs = new_dirs / new_dirs.norm(dim=1, keepdim=True)
                
                # Update weights
                self.W_enc.data[:, to_prune] = new_dirs.T * self.W_enc.data[:, ~to_prune].norm(dim=0).mean()
                self.W_dec.data[to_prune, :] = new_dirs
                self.b_enc.data[to_prune] = 0.0
                
                # Reset importance scores for pruned neurons
                self.feature_importance[to_prune] = self.feature_importance.mean()

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
class CustomTrainer(SAETrainer):
    """Trainer for Custom Sparse Autoencoder using L1 regularization."""
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
        self.ae = CustomSAE(d_in=activation_dim, d_sae=dict_size, hook_layer=layer, model_name=lm_name)

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
        
        # Reconstruction loss
        l2_loss = torch.linalg.norm(x - x_hat, dim=-1).mean()
        
        # Sparsity loss
        l1_loss = f.norm(p=1, dim=-1).mean()
        
        # Orthogonality loss for top pairs
        row_idx, col_idx, values = self._get_top_pairs(f)
        ortho_loss = torch.mean(values) * self.tau

        if self.steps_since_active is not None:
            # Update steps_since_active
            deads = (f == 0).all(dim=0)
            self.steps_since_active[deads] += 1
            self.steps_since_active[~deads] = 0
        
        loss = l2_loss + self.l1_penalty * l1_loss + ortho_loss

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
    trainer = CustomTrainer(
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
    """Convert string dtype to torch dtype."""
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
        print(f"Completed training SAE for layer {layer}")
