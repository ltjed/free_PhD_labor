import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import torch
import torch.nn as nn
import numpy as np
from collections import namedtuple
from huggingface_hub import hf_hub_download
from dataclasses import dataclass
from typing import Optional, Tuple
from datetime import datetime
from nnsight import LanguageModel
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.buffer import ActivationBuffer
import argparse
import time
from torch.autograd import Function

import einops
import torch as t
import torch.nn as nn
from collections import namedtuple

from dictionary_learning.config import DEBUG
from dictionary_learning.dictionary import Dictionary
from dictionary_learning.trainers.trainer import SAETrainer


@t.no_grad()
def geometric_median(points: t.Tensor, max_iter: int = 100, tol: float = 1e-5):
    """Compute the geometric median `points`. Used for initializing decoder bias."""
    # Initialize our guess as the mean of the points
    guess = points.mean(dim=0)
    prev = t.zeros_like(guess)

    # Weights for iteratively reweighted least squares
    weights = t.ones(len(points), device=points.device)

    for _ in range(max_iter):
        prev = guess

        # Compute the weights
        weights = 1 / t.norm(points - guess, dim=1)

        # Normalize the weights
        weights /= weights.sum()

        # Compute the new geometric median
        guess = (weights.unsqueeze(1) * points).sum(dim=0)

        # Early stopping condition
        if t.norm(guess - prev) < tol:
            break

    return guess


class AutoEncoderTopK(nn.Module):
    """
    The top-k autoencoder architecture using parameters instead of nn.Linear layers.
    """
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        hook_layer: int,
        model_name: str = "EleutherAI/pythia-70m-deduped",
        hook_name: Optional[str] = None,
        k: int = 100,
    ):
        super().__init__()
        self.activation_dim = d_in
        self.dict_size = d_sae
        self.k = k

        # Initialize encoder parameters
        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        nn.init.kaiming_uniform_(self.W_enc, nonlinearity='relu')
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        
        # Initialize decoder parameters (transposed and normalized)
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
        self.W_dec.data = self.W_enc.data.T.clone()
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.set_decoder_norm_to_unit_norm()


        if hook_name is None:
            hook_name = f"blocks.{hook_layer}.hook_resid_post"
        # Configuration
        self.cfg = CustomSAEConfig(
            model_name=model_name,
            d_in=d_in,
            d_sae=d_sae,
            hook_name=hook_name,
            hook_layer=hook_layer,
            architecture="TopK",
            activation_fn_str="TopK",
            apply_b_dec_to_input=True,
        )
        

    def compute_competition_coefficients(self, top_indices_BK):
        """Compute directional competition coefficients between features"""
        B = top_indices_BK.size(0)
        F = self.dict_size
        k = self.k
        
        # Convert indices to binary masks
        masks = torch.zeros((B, F), device=top_indices_BK.device)
        masks.scatter_(1, top_indices_BK, 1.0)
        
        # Compute co-activation counts
        coact_counts = torch.mm(masks.t(), masks)  # F x F
        
        # Get individual activation counts
        act_counts = coact_counts.diag().unsqueeze(1)  # F x 1
        
        # Compute conditional probabilities n_i|j/n_j
        cond_probs = coact_counts / (act_counts.t() + 1e-8)  # F x F
        
        # Normalize by sparsity ratio k/d to get competition coefficients
        sparsity_ratio = k / F
        competition_coeffs = cond_probs / sparsity_ratio
        
        return competition_coeffs

    def encode(self, x: torch.Tensor, return_topk: bool = False):
        pre_acts = (x - self.b_dec) @ self.W_enc + self.b_enc
        post_relu_feat_acts_BF = torch.relu(pre_acts)
        post_topk = post_relu_feat_acts_BF.topk(self.k, sorted=False, dim=-1)

        # Scatter topk values to form encoded activations
        tops_acts_BK, top_indices_BK = post_topk.values, post_topk.indices
        encoded_acts_BF = torch.zeros_like(post_relu_feat_acts_BF).scatter_(
            dim=-1, index=top_indices_BK, src=tops_acts_BK
        )

        if return_topk:
            return encoded_acts_BF, tops_acts_BK, top_indices_BK
        else:
            return encoded_acts_BF

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor, output_features: bool = False):
        encoded_acts = self.encode(x)
        x_hat = self.decode(encoded_acts)
        return (x_hat, encoded_acts) if output_features else x_hat

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device = kwargs.get("device", None)
        dtype = kwargs.get("dtype", None)
        if device:
            self.device = device
        if dtype:
            self.dtype = dtype
        return self

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        eps = torch.finfo(self.W_dec.dtype).eps
        norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data /= norm + eps

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        if self.W_dec.grad is None:
            return

        # Compute parallel component for each dictionary element (rows of W_dec)
        parallel_component = torch.einsum(
            'sd,sd->s', 
            self.W_dec.grad, 
            self.W_dec.data
        )
        # Subtract parallel component from gradient
        self.W_dec.grad -= torch.einsum(
            's,sd->sd', 
            parallel_component, 
            self.W_dec.data
        )

    @classmethod
    def from_pretrained(cls, path: str, k: int, device=None):
        state_dict = torch.load(path)
        # Original encoder weight: (d_sae, d_in) -> Transpose to (d_in, d_sae)
        d_sae, d_in = state_dict["encoder.weight"].shape
        autoencoder = cls(d_in, d_sae, hook_layer=0, k=k)  # Default hook_layer
        
        # Map original state_dict to new parameter names
        new_state_dict = {
            'W_enc': state_dict['encoder.weight'].T,
            'b_enc': state_dict['encoder.bias'],
            'W_dec': state_dict['decoder.weight'].T,
            'b_dec': state_dict['b_dec']
        }
        autoencoder.load_state_dict(new_state_dict)
        
        if device is not None:
            autoencoder.to(device)
        return autoencoder


class TrainerTopK(SAETrainer):
    """
    Top-K SAE training scheme.
    """

    def __init__(
        self,
        dict_class=AutoEncoderTopK,
        activation_dim=512,
        dict_size=64 * 512,
        k=100,
        auxk_alpha=1 / 32,  # see Appendix A.2
        ortho_weight=0.01,  # weight for orthogonality loss
        decay_start=24000,  # when does the lr decay start
        steps=30000,  # when when does training end
        seed=None,
        device=None,
        layer=None,
        lm_name=None,
        wandb_name="AutoEncoderTopK",
        submodule_name=None,
    ):
        super().__init__(seed)

        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name

        self.wandb_name = wandb_name
        self.steps = steps
        self.k = k
        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        # Initialise autoencoder
        self.ae =  AutoEncoderTopK(
            d_in=activation_dim,
            d_sae=dict_size,
            hook_layer=layer,
            model_name=lm_name,
            k = k
            )


        if device is None:
            self.device = "cuda" if t.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.ae.to(self.device)

        # Auto-select LR using 1 / sqrt(d) scaling law from Figure 3 of the paper
        scale = dict_size / (2**14)
        self.lr = 2e-4 / scale**0.5
        self.auxk_alpha = auxk_alpha
        self.ortho_weight = ortho_weight
        self.dead_feature_threshold = 10_000_000

        # Optimizer and scheduler
        self.optimizer = t.optim.Adam(self.ae.parameters(), lr=self.lr, betas=(0.9, 0.999))

        def lr_fn(step):
            if step < decay_start:
                return 1.0
            else:
                return (steps - step) / (steps - decay_start)

        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

        # Training parameters
        self.num_tokens_since_fired = t.zeros(dict_size, dtype=t.long, device=device)

        # Log the effective L0, i.e. number of features actually used, which should a constant value (K)
        # Note: The standard L0 is essentially a measure of dead features for Top-K SAEs)
        self.logging_parameters = ["effective_l0", "dead_features"]
        self.effective_l0 = -1
        self.dead_features = -1

    def loss(self, x, step=None, logging=False):
        # Run the SAE
        f, top_acts, top_indices = self.ae.encode(x, return_topk=True)
        x_hat = self.ae.decode(f)

        # Measure goodness of reconstruction
        e = x_hat - x
        total_variance = (x - x.mean(0)).pow(2).sum(0)

        # Update the effective L0 (again, should just be K)
        self.effective_l0 = top_acts.size(1)

        # Update "number of tokens since fired" for each features
        num_tokens_in_step = x.size(0)
        did_fire = t.zeros_like(self.num_tokens_since_fired, dtype=t.bool)
        did_fire[top_indices.flatten()] = True
        self.num_tokens_since_fired += num_tokens_in_step
        self.num_tokens_since_fired[did_fire] = 0

        # Compute dead feature mask based on "number of tokens since fired"
        dead_mask = (
            self.num_tokens_since_fired > self.dead_feature_threshold
            if self.auxk_alpha > 0
            else None
        ).to(f.device)
        self.dead_features = int(dead_mask.sum())

        # If dead features: Second decoder pass for AuxK loss
        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            # Heuristic from Appendix B.1 in the paper
            k_aux = x.shape[-1] // 2

            # Reduce the scale of the loss if there are a small number of dead latents
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)

            # Don't include living latents in this loss
            auxk_latents = t.where(dead_mask[None], f, -t.inf)

            # Top-k dead latents
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

            auxk_buffer_BF = t.zeros_like(f)
            auxk_acts_BF = auxk_buffer_BF.scatter_(dim=-1, index=auxk_indices, src=auxk_acts)

            # Encourage the top ~50% of dead latents to predict the residual of the
            # top k living latents
            e_hat = self.ae.decode(auxk_acts_BF)
            auxk_loss = (e_hat - e).pow(2)  # .sum(0)
            auxk_loss = scale * t.mean(auxk_loss / total_variance)
        else:
            auxk_loss = x_hat.new_tensor(0.0)

        # Compute hierarchical competition loss
        competition_coeffs = self.ae.compute_competition_coefficients(top_indices)
        feature_dots = t.einsum('bi,bj->ij', f, f) / x.size(0)
        
        # Take max of directional coefficients to penalize both directions
        max_coeffs = torch.maximum(competition_coeffs, competition_coeffs.t())
        # Clip competition coefficients for stability
        competition_loss = (max_coeffs.clamp(max=10.0) * feature_dots.pow(2)).mean()

        l2_loss = e.pow(2).sum(dim=-1).mean()
        auxk_loss = auxk_loss.sum(dim=-1).mean()
        loss = l2_loss + self.auxk_alpha * auxk_loss + self.ortho_weight * competition_loss

        if not logging:
            return loss
        else:
            return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
                x,
                x_hat,
                f,
                {"l2_loss": l2_loss.item(), "auxk_loss": auxk_loss.item(), "loss": loss.item()},
            )

    def update(self, step, x):
        # Initialise the decoder bias
        if step == 0:
            median = geometric_median(x)
            self.ae.b_dec.data = median

        # Make sure the decoder is still unit-norm
        self.ae.set_decoder_norm_to_unit_norm()

        # compute the loss
        x = x.to(self.device)
        loss = self.loss(x, step=step)
        loss.backward()

        # clip grad norm and remove grads parallel to decoder directions
        t.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
        self.ae.remove_gradient_parallel_to_decoder_directions()

        # do a training step
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        return loss.item()

    @property
    def config(self):
        return {
            "trainer_class": "TrainerTopK",
            "dict_class": "AutoEncoderTopK",
            "lr": self.lr,
            "steps": self.steps,
            "seed": self.seed,
            "activation_dim": self.ae.activation_dim,
            "dict_size": self.ae.dict_size,
            "k": self.ae.k,
            "device": self.device,
            "layer": self.layer,
            "lm_name": self.lm_name,
            "wandb_name": self.wandb_name,
            "submodule_name": self.submodule_name,
            "ortho_weight": self.ortho_weight,
        }


# Do not modify CustomSAEConfig class as this defines the right format for SAE to be evaluated!
@dataclass
class CustomSAEConfig:
    model_name: str
    d_in: int
    d_sae: int
    hook_layer: int
    hook_name: str

    context_size: int = None
    hook_head_index: Optional[int] = None
    architecture: str = ""
    apply_b_dec_to_input: bool = None
    finetuning_scaling_factor: bool = None
    activation_fn_str: str = ""
    activation_fn_kwargs = {}
    prepend_bos: bool = True
    normalize_activations: str = "none"
    dtype: str = ""
    device: str = ""
    model_from_pretrained_kwargs = {}
    dataset_path: str = ""
    dataset_trust_remote_code: bool = True
    seqpos_slice: tuple = (None,)
    training_tokens: int = -100_000
    sae_lens_training_version: Optional[str] = None
    neuronpedia_id: Optional[str] = None






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


def special_slice(list1, m=5, num_k=20):
    # Compute N
    N = (len(list1) - m) // num_k

    # Create dictionary with step indices
    result = {}
    for k in range(num_k):
        for i in range(m):
            idx = k * N + i
            if idx < len(list1):
                result[f"step {idx}"] = list1[idx]
    
    return result

def run_sae_training(
    layer: int,
    dict_size: int,
    num_tokens: int,
    out_dir: str,
    device: str,
    model_name: str = "google/gemma-2-2b",
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
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)


    # Calculate steps
    steps = int(num_tokens / sae_batch_size)


    model = LanguageModel(
        model_name,
        device_map=device,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        cache_dir=None,
    )
    
    if model_name == "EleutherAI/pythia-70m-deduped":
        submodule = model.gpt_neox.layers[layer]
    else:
        submodule = model.model.layers[layer]
    submodule_name = f"resid_post_layer_{layer}"
    activation_dim = model.config.hidden_size

    # Configure dataset loading with retries and increased timeout
    from huggingface_hub import HfApi
    import time
    from functools import partial
    from datasets import load_dataset
    
    max_retries = 5
    retry_delay = 10
    timeout = 60
    
    def load_with_retry():
        dataset = load_dataset("monology/pile-uncopyrighted", streaming=True)
        return dataset['train']
    
    for attempt in range(max_retries):
        try:
            dataset = load_with_retry()
            generator = (item['text'] for item in dataset)
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Failed to load dataset after all retries")
                raise
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
    trainer = TrainerTopK(
        activation_dim=activation_dim,
        dict_class=AutoEncoderTopK,
        dict_size=dict_size,
        k=120, 
        auxk_alpha = 1/32,
        ortho_weight= 0.01,  # Reduced from 0.1
        decay_start=steps/8*7,
        steps = steps,
        seed=seed,
        device=device,
        layer=layer,
        lm_name=model_name,
        submodule_name=submodule_name,
    )
    training_log = []
    for step in range(steps):
        activations = next(activation_buffer)
        loss_dict = trainer.update(step, activations)
        training_log.append(loss_dict)

        if step % 100 == 0:
            print(f"Step {step}: {loss_dict}")
    
    
    
    
    print("\n training complete! \n")
    # Prepare final results
    final_info = {
        "training_steps": steps,
        "final_loss": training_log[-1] if training_log else None,
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
        # "training_log": special_slice(training_log, m = 2, num_k= 4),
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
    existing_data.update({f"training results for layer {layer}" : results})
    with open(all_info_path, "w") as f:
        json.dump(existing_data, indent=2, fp=f)
    print(f"all info: {all_info_path}") 
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
    # "EleutherAI/pythia-70m-deduped": {"batch_size": 512, "dtype": "float32", "layers": [3], "d_model": 512},
    # "google/gemma-2-2b": {"batch_size": 32, "dtype": "bfloat16", "layers": [5, 12, 19], "d_model": 2304},
    "google/gemma-2-2b": {"batch_size": 32, "dtype": "bfloat16", "layers": [12], "d_model": 4608},
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
                    perform_scr = False,
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
    
    # Create required directories                                                                                                                                                                                                    
    import os
    import shutil

    # List of directories to be created
    directories = [
        "artifacts/autointerp/google",
        "artifacts/absorption",
        "artifacts/scr",
        "artifacts/tpp",
        "artifacts/sparse_probing"
    ]

    for dir_path in directories:
        # If directory exists, delete it and its contents
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        
        # Recreate the directory
        os.makedirs(dir_path)
    
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
    d_model = 18432  # Return to optimal dictionary size
    llm_batch_size = MODEL_CONFIGS[model_name]["batch_size"]
    llm_dtype = MODEL_CONFIGS[model_name]["dtype"]
    # Initialize variables that were previously args
    layers = MODEL_CONFIGS[model_name]["layers"]
    num_tokens = 5_000_000 # Set default number of tokens, can be increased by a factor of up to 10 but takes much longer. Note training steps = num_tokens/sae_batch_size, so you can increase training be increasing num_of_tokens
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
    wandb_project=None,
    )) 



    device = "cuda" if torch.cuda.is_available() else "cpu"


    # Note: Unlearning is not recommended for models with < 2B parameters and we recommend an instruct tuned model
    # Unlearning will also require requesting permission for the WMDP dataset (see unlearning/README.md)
    # Absorption not recommended for models < 2B parameters

    # Select your eval types here.

    # "autointerp", AUTOINTERP NOT AVAILABLE
    # "unlearning", UNLEARNING CURRENTLY UNAVAILABLE

    eval_types = [
        "absorption", # LOWER ABSORPTION SCORE MEANS BETTER PERFORMANCE
        # "autointerp",
        "core",
        "scr",
        "tpp",
        "sparse_probing",
        "unlearning",
    ]

    if "autointerp" in eval_types:
        try:
            api_key = os.environ["OPENAI_API_KEY"]
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
            force_rerun=True,
            save_activations=False,
            out_dir=save_dir
        )
