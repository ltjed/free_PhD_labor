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
    """Implementation of Self-Correcting Orthogonal SAE with Adaptive Thresholding."""
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        hook_layer: int,
        model_name: str = "EleutherAI/pythia-70m-deduped",
        hook_name: Optional[str] = None,
        ema_decay: float = 0.99,
        z_threshold: float = 1.5,
        ortho_penalty: float = 0.01,
        svd_rank: int = 32,
        skip_scale: float = 0.1,  # Skip connection scaling factor
        norm_eps: float = 1e-5
    ):
        super().__init__()
        # Initialize with scaled random weights
        self.W_enc = nn.Parameter(torch.randn(d_in, d_sae) / np.sqrt(d_in))
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_in) / np.sqrt(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self.skip_scale = skip_scale
        self.norm_eps = norm_eps
        
        # Adaptive layer normalization parameters
        self.ln_scale = nn.Parameter(torch.ones(d_in))
        self.ln_bias = nn.Parameter(torch.zeros(d_in))
        self.adaptive_scale = nn.Parameter(torch.ones(1))
        
        # EMA tracking
        self.register_buffer('ema_mean', torch.zeros(d_sae))
        self.register_buffer('ema_var', torch.ones(d_sae))
        self.register_buffer('ema_cov', torch.zeros(d_sae, d_sae))
        self.ema_decay = ema_decay
        self.z_threshold = z_threshold
        self.ortho_penalty = ortho_penalty
        self.svd_rank = svd_rank
        
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
        # Apply adaptive layer normalization
        mean = input_acts.mean(dim=-1, keepdim=True)
        var = input_acts.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (input_acts - mean) / torch.sqrt(var + self.norm_eps)
        # Apply learned adaptive scaling
        scale_factor = torch.sigmoid(self.adaptive_scale)
        normalized = normalized * (self.ln_scale * scale_factor) + self.ln_bias
        
        pre_acts = (normalized - self.b_dec) @ self.W_enc + self.b_enc
        
        # Update EMA statistics
        with torch.no_grad():
            batch_mean = pre_acts.mean(0)
            batch_var = pre_acts.var(0, unbiased=False)
            self.ema_mean = self.ema_decay * self.ema_mean + (1 - self.ema_decay) * batch_mean
            self.ema_var = self.ema_decay * self.ema_var + (1 - self.ema_decay) * batch_var
            
            # Compute batch covariance using randomized SVD for efficiency
            centered = (pre_acts - batch_mean).to(torch.float32)
            batch_cov = torch.matmul(centered.transpose(-2, -1), centered) / centered.size(0)
            batch_cov = batch_cov.to(pre_acts.dtype)
            
            # Update EMA covariance
            self.ema_cov = self.ema_decay * self.ema_cov + (1 - self.ema_decay) * batch_cov
        
        # Apply adaptive z-score thresholding
        z_scores = (pre_acts - self.ema_mean) / torch.sqrt(self.ema_var + 1e-5)
        mask = (z_scores.abs() > self.z_threshold)
        acts = torch.relu(pre_acts * mask)
        return acts

    def decode(self, acts, input_acts=None):
        decoded = (acts @ self.W_dec) + self.b_dec
        if input_acts is not None:
            decoded = decoded + self.skip_scale * input_acts
        return decoded

    def forward(self, acts, output_features=False):
        encoded = self.encode(acts)
        decoded = self.decode(encoded, acts)
        
        # Compute orthogonality penalty
        ortho_loss = torch.norm(
            torch.matmul(self.W_enc.T, self.W_enc) - 
            torch.eye(self.W_enc.shape[1], device=self.device)
        )
        
        if output_features:
            return decoded, encoded, ortho_loss * self.ortho_penalty
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
    """A variant of Adam with gradient clipping, scaled normalization and adaptive learning rate."""
    def __init__(self, params, constrained_params, lr, max_grad_norm=1.0, min_lr=1e-5):
        super().__init__(params, lr=lr)
        self.constrained_params = list(constrained_params)
        self.max_grad_norm = max_grad_norm
        self.min_lr = min_lr
        self.initial_lr = lr
        self.steps = 0
        
    def step(self, closure=None):
        self.steps += 1
        
        # Adaptive learning rate based on gradient statistics
        grad_norm = torch.nn.utils.clip_grad_norm_(self.param_groups[0]['params'], self.max_grad_norm)
        if grad_norm > self.max_grad_norm:
            self.param_groups[0]['lr'] = max(
                self.min_lr,
                self.param_groups[0]['lr'] * 0.9
            )
        elif self.steps % 100 == 0:
            self.param_groups[0]['lr'] = min(
                self.initial_lr,
                self.param_groups[0]['lr'] * 1.1
            )
        
        with torch.no_grad():
            for p in self.constrained_params:
                if p.grad is not None:
                    normed_p = p / (p.norm(dim=0, keepdim=True) + 1e-6)
                    p.grad -= (p.grad * normed_p).sum(dim=0, keepdim=True) * normed_p
        
        super().step(closure=closure)
        
        with torch.no_grad():
            for p in self.constrained_params:
                p.data /= (p.data.norm(dim=0, keepdim=True) + 1e-6)


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
    """Trainer for Self-Correcting Orthogonal SAE with Adaptive Thresholding."""
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
        self.ae = OrthogonalSAE(
            d_in=activation_dim, 
            d_sae=dict_size, 
            hook_layer=layer, 
            model_name=lm_name,
            ema_decay=0.99,
            z_threshold=2.0,
            ortho_penalty=0.01,  # Reduced penalty
            svd_rank=32
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
        x_hat, f, ortho_loss = self.ae(x, output_features=True)
        
        # Compute per-feature reconstruction contribution
        with torch.no_grad():
            feature_importance = torch.zeros(self.ae.dict_size, device=x.device)
            for i in range(self.ae.dict_size):
                mask = torch.zeros_like(f)
                mask[:, i] = f[:, i]
                feat_reconstruction = self.ae.decode(mask)
                feat_contribution = torch.linalg.norm(feat_reconstruction, dim=-1).mean()
                feature_importance[i] = feat_contribution
            
            # Normalize importance scores
            feature_importance = (feature_importance - feature_importance.mean()) / (feature_importance.std() + 1e-8)
            importance_weights = torch.sigmoid(feature_importance)
            
        # Weighted losses
        l2_loss = torch.linalg.norm(x - x_hat, dim=-1).mean()
        l1_loss = (f.norm(p=1, dim=0) * importance_weights).mean()

        if self.steps_since_active is not None:
            deads = (f == 0).all(dim=0)
            self.steps_since_active[deads] += 1
            self.steps_since_active[~deads] = 0
        
        loss = l2_loss + self.l1_penalty * l1_loss + ortho_loss

        # Store importance weights for gradient scaling
        self.importance_weights = importance_weights

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
                    'mean_importance': importance_weights.mean().item()
                }
            )

    def update(self, step, activations):
        activations = activations.to(self.device)

        self.optimizer.zero_grad()
        loss = self.loss(activations)
        loss.backward()
        
        # Scale gradients based on feature importance
        with torch.no_grad():
            if hasattr(self, 'importance_weights'):
                for param in self.ae.parameters():
                    if param.grad is not None and param.size(0) == self.ae.dict_size:
                        param.grad *= self.importance_weights.unsqueeze(-1)
        
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
    trainer = OrthogonalTrainer(
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
            learning_rate=3e-4,
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
