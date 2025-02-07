import sys
import os
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import torch
import torch.nn as nn
import numpy as np
from collections import namedtuple
import requests
from requests.adapters import HTTPAdapter 
from requests.packages.urllib3.util.retry import Retry
from huggingface_hub.utils import HfHubHTTPError
import time

from dataclasses import dataclass
from typing import Optional, Tuple
from datetime import datetime
from nnsight import LanguageModel
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.buffer import ActivationBuffer
import argparse
import time
from torch.autograd import Function

from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

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
    jump_coeff: float = 0.1

class JumpReLUFunc(Function):
    @staticmethod
    def forward(ctx, pre_acts, jump_coeff):
        ctx.save_for_backward(pre_acts)
        ctx.jump_coeff = jump_coeff
        jump_term = jump_coeff * (pre_acts > 0).to(dtype=pre_acts.dtype)
        return torch.relu(pre_acts) + jump_term

    @staticmethod
    def backward(ctx, grad_output):
        pre_acts, = ctx.saved_tensors
        jump_coeff = ctx.jump_coeff
        mask = (pre_acts > 0).to(dtype=pre_acts.dtype)
        grad_input = grad_output * (1 + jump_coeff) * mask
        return grad_input, None

class JumpReLUSAE(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        hook_layer: int,
        model_name: str = "EleutherAI/pythia-70m-deduped",
        hook_name: Optional[str] = None,
        jump_coeff: float = 0.1,
    ):
        # Initialize previous activation state
        self.prev_acts = None
        super().__init__()
        self.W_enc = nn.Parameter(torch.zeros(d_in, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        nn.init.kaiming_uniform_(self.W_enc, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.W_dec, nonlinearity='relu')
        nn.init.uniform_(self.b_enc, -0.01, 0.01)
        nn.init.uniform_(self.b_dec, -0.01, 0.01)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.activation_dim = d_in
        self.dict_size = d_sae
        self.dtype = torch.float32
        
        if hook_name is None:
            hook_name = f"blocks.{hook_layer}.hook_resid_post"

        self.cfg = CustomSAEConfig(
            model_name=model_name,
            d_in=d_in,
            d_sae=d_sae,
            hook_name=hook_name,
            hook_layer=hook_layer,
            architecture="JumpReLU",
            activation_fn_str="jumprelu",
            apply_b_dec_to_input=True,
            jump_coeff=jump_coeff,
        )

    def encode(self, input_acts):
        pre_acts = (input_acts - self.b_dec) @ self.W_enc + self.b_enc
        acts = JumpReLUFunc.apply(pre_acts, self.cfg.jump_coeff)
        # Store current activation for next step
        if self.prev_acts is None:
            self.prev_acts = acts.detach()
        return acts

    def decode(self, acts):
        return (acts @ self.W_dec) + self.b_dec

    def forward(self, acts, output_features=False):
        encoded = self.encode(acts)
        decoded = self.decode(encoded)
        return (decoded, encoded) if output_features else decoded

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        return self

def set_decoder_norm_to_unit_norm(decoder_weight, activation_dim, dict_size):
    with torch.no_grad():
        decoder_weight.data = decoder_weight / decoder_weight.norm(dim=0, keepdim=True)
    return decoder_weight

def remove_gradient_parallel_to_decoder_directions(decoder_weight, grad, activation_dim, dict_size):
    parallel_component = torch.einsum('ij,ij->j', grad, decoder_weight)
    grad -= parallel_component.unsqueeze(0) * decoder_weight
    return grad

class SAETrainer:
    def __init__(self, seed=None):
        self.seed = seed
        self.logging_parameters = []

    def update(self, step, activations):
        pass

    def get_logging_parameters(self):
        return {param: getattr(self, param) for param in self.logging_parameters if hasattr(self, param)}

    @property
    def config(self):
        return {'wandb_name': 'trainer'}

class JumpReLUTrainer(SAETrainer):
    def __init__(
        self,
        activation_dim=512,
        dict_size=64*512,
        lr=7e-5,
        l1_penalty=1e-1,
        warmup_steps=1000,
        resample_steps=None,
        steps = None,
        seed=None,
        device=None,
        layer=None,
        lm_name=None,
        wandb_name='JumpReLUTrainer',
        submodule_name=None,
        jump_coeff: float = 0.1,
        bandwidth: float = 0.001,
        sparsity_penalty: float = 1.0,
        sparsity_warmup_steps: int = 2000,
        target_l0: float = 20.0,
    ):
        super().__init__(seed)
        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.resample_steps = int(steps * 0.3)
        self.resample_counter = 0
        torch.manual_seed(seed or 42)
        self.device = device or 'cuda' if torch.cuda.is_available() else 'cpu'
        self.ae = JumpReLUSAE(
            d_in=activation_dim,
            d_sae=dict_size,
            hook_layer=layer,
            model_name=lm_name,
            jump_coeff=jump_coeff,
        ).to(self.device)

        self.lr = lr
        self.sparsity_penalty = sparsity_penalty
        self.warmup_steps = warmup_steps
        self.sparsity_warmup_steps = sparsity_warmup_steps
        self.target_l0 = target_l0
        self.bandwidth = bandwidth
        self.resample_steps = resample_steps

        self.optimizer = Adam(
            self.ae.parameters(),
            lr=lr,
            betas=(0.0, 0.999),
            eps=1e-8
        )

        self.lr_scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(step / warmup_steps, 1.0)
        )

        self.sparsity_warmup_fn = lambda step: min(step / sparsity_warmup_steps, 1.0)
        self.num_tokens_since_fired = torch.zeros(dict_size, device=self.device)
        self.logging_parameters.extend(['dead_features'])

    def loss(self, x, step, logging=False):
        x = x.to(self.ae.W_enc.dtype)
        pre_acts = (x - self.ae.b_dec) @ self.ae.W_enc + self.ae.b_enc
        f = JumpReLUFunc.apply(pre_acts, self.ae.cfg.jump_coeff)
        
        active_neurons = (f > 0).any(dim=0)
        self.num_tokens_since_fired[active_neurons] = 0
        self.num_tokens_since_fired[~active_neurons] += x.size(0)
        dead_features = (self.num_tokens_since_fired > 1e6).sum().item()

        recon = self.ae.decode(f)
        recon_loss = (x - recon).pow(2).sum(dim=-1).mean()
        l0 = (pre_acts > 0).float().sum(dim=-1).mean()
        
        # Initialize temporal coefficient and loss
        temporal_loss = 0.0
        temporal_coeff = 0.0
        
        if self.ae.prev_acts is not None:
            # Sigmoid-based temporal coefficient scaling
            progress = torch.tensor(step / self.warmup_steps, device=f.device)
            temporal_coeff = 0.05 + 0.25 / (1 + torch.exp(-10 * (progress - 0.5)))
            temporal_loss = (f - self.ae.prev_acts).abs().mean()
        self.ae.prev_acts = f.detach()
        
        sparsity_scale = self.sparsity_warmup_fn(step)
        sparsity_loss = self.sparsity_penalty * ((l0 / self.target_l0) - 1).pow(2) * sparsity_scale
        total_loss = recon_loss + sparsity_loss + temporal_coeff * temporal_loss

        if logging:
            return {
                "loss": total_loss.item(),
                "mse_loss": recon_loss.item(),
                "sparsity_loss": sparsity_loss.item(),
                "temporal_loss": temporal_loss.item() if isinstance(temporal_loss, torch.Tensor) else temporal_loss,
                "l0": l0.item(),
                "dead_features": dead_features
            }
        return total_loss

    def update(self, step, activations):
        activations = activations.to(self.device)
        self.optimizer.zero_grad()
        loss = self.loss(activations, step)
        loss.backward()

        with torch.no_grad():
            self.ae.W_dec.grad = remove_gradient_parallel_to_decoder_directions(
                self.ae.W_dec.T,
                self.ae.W_dec.grad.T,
                self.ae.activation_dim,
                self.ae.dict_size
            ).T

        torch.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
        self.optimizer.step()
        self.lr_scheduler.step()

        with torch.no_grad():
            self.ae.W_dec.data = set_decoder_norm_to_unit_norm(
                self.ae.W_dec.T,
                self.ae.activation_dim,
                self.ae.dict_size
            ).T

        if self.resample_steps and step % self.resample_steps == 0:
            print("\n\n\nRESAMPLEING\n\n\n")
            self.resample_neurons(activations)
            self.resample_counter += 1
            if (self.resample_counter == 2):
                self.resample_counter = 0
                self.resample_steps = None

        return self.loss(activations, step, logging=True)

    def resample_neurons(self, activations):
        dead_mask = self.num_tokens_since_fired > self.resample_steps
        if not dead_mask.any():
            return
        print("\n\n\n{dead_mask.sum()}\n\n\n")
        losses = (activations - self.ae(activations)).norm(dim=-1)
        n_resample = min(dead_mask.sum().item(), losses.size(0))
        indices = torch.multinomial(losses, n_resample, replacement=False)
        sampled = activations[indices]

        with torch.no_grad():
            alive_norm = self.ae.W_enc[:, ~dead_mask].norm(dim=0).mean()
            self.ae.W_enc[:, dead_mask] = sampled.T * alive_norm * 0.2
            self.ae.b_enc[dead_mask] = 0
            self.ae.W_dec[dead_mask] = sampled / sampled.norm(dim=1, keepdim=True)

            self.optimizer.state_dict()['state'][self.ae.W_enc]['exp_avg'][:, dead_mask] = 0
            self.optimizer.state_dict()['state'][self.ae.W_enc]['exp_avg_sq'][:, dead_mask] = 0

    @property
    def config(self):
        return {
            'trainer_class': 'JumpReLUTrainer',
            'activation_dim': self.ae.activation_dim,
            'dict_size': self.ae.dict_size,
            'lr': self.lr,
            'sparsity_penalty': self.sparsity_penalty,
            'warmup_steps': self.warmup_steps,
            'resample_steps': self.resample_steps,
            'device': self.device,
            'layer': self.layer,
            'lm_name': self.lm_name,
            'wandb_name': 'JumpReLUTrainer',
            'jump_coeff': self.ae.cfg.jump_coeff,
            'target_l0': self.target_l0,
            'bandwidth': self.bandwidth
        }

def special_slice(list1, m=5, num_k=20):
    N = (len(list1) - m) // num_k
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
    jump_coeff: float = 0.1,  # New parameter
):
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)


    # Calculate steps
    steps = int(num_tokens / sae_batch_size)


    # Configure aggressive retries and timeouts for requests
    session = requests.Session()
    retries = Retry(
        total=10,
        backoff_factor=2,
        status_forcelist=[408, 429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"]
    )
    adapter = HTTPAdapter(max_retries=retries, pool_maxsize=100)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    session.timeout = 300  # 5 minute timeout
    
    # Initialize model with increased timeout
    model = LanguageModel(
        model_name,
        device_map=device,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        cache_dir=None
    )
    
    if model_name == "EleutherAI/pythia-70m-deduped":
        submodule = model.gpt_neox.layers[layer]
    else:
        submodule = model.model.layers[layer]
    submodule_name = f"resid_post_layer_{layer}"
    activation_dim = model.config.hidden_size

    # Configure more aggressive retries for dataset downloads
    session = requests.Session()
    retries = Retry(
        total=10,
        backoff_factor=2,
        status_forcelist=[408, 429, 500, 502, 503, 504],
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))
    
    # Try multiple times to get dataset with increased timeout
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            generator = hf_dataset_to_generator(
                "monology/pile-uncopyrighted"
            )
            break
        except (requests.exceptions.Timeout, HfHubHTTPError) as e:
            if attempt == max_attempts - 1:
                raise
            print(f"Dataset download attempt {attempt + 1} failed, retrying...")
            time.sleep(5 * (attempt + 1))
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
    trainer = JumpReLUTrainer(
        activation_dim=activation_dim,
        dict_size=dict_size,
        lr=learning_rate,
        l1_penalty=sparsity_penalty,
        steps = steps,
        warmup_steps=warmup_steps,
        seed=seed,
        device=device,
        layer=layer,
        lm_name=model_name,
        submodule_name=submodule_name,
        jump_coeff=jump_coeff,
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
    "google/gemma-2-2b": {"batch_size": 32, "dtype": "bfloat16", "layers": [12], "d_model": 65536},
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
        "scr_and_tpp": (
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
    
    directories = [
        "artifacts/autointerp/google",
        "artifacts/absorption",
        "artifacts/scr_and_tpp",
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
    d_model = MODEL_CONFIGS[model_name]["d_model"]
    llm_batch_size = MODEL_CONFIGS[model_name]["batch_size"]
    llm_dtype = MODEL_CONFIGS[model_name]["dtype"]
    # Initialize variables that were previously args
    layers = MODEL_CONFIGS[model_name]["layers"]
    num_tokens = 10_000_000 # Set default number of tokens, can be increased by a factor of up to 10 but takes much longer. Note training steps = num_tokens/sae_batch_size, so you can increase training be increasing num_of_tokens
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
    jump_coeff=0.1
    )) 



    device = "cuda" if torch.cuda.is_available() else "cpu"


    # Note: Unlearning is not recommended for models with < 2B parameters and we recommend an instruct tuned model
    # Unlearning will also require requesting permission for the WMDP dataset (see unlearning/README.md)
    # Absorption not recommended for models < 2B parameters

    # Select your eval types here.

    # "autointerp", AUTOINTERP NOT AVAILABLE
    # "unlearning", UNLEARNING CURRENTLY UNAVAILABLE

    eval_types = [
        "absorption",
        # "autointerp",
        "core",
        "scr_and_tpp",
        "sparse_probing",
        # "unlearning",
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
            force_rerun=False,
            save_activations=False,
            out_dir=save_dir
        )
