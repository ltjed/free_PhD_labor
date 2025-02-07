import sys
import os
import shutil
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

class DualTemporalSAE(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        hook_layer: int,
        model_name: str = "EleutherAI/pythia-70m-deduped",
        hook_name: Optional[str] = None,
        context_size: int = 8,
        lambda1: float = 0.6,
        lambda2: float = 0.4,
        alpha1: float = 0.03,
        alpha2: float = 0.06
    ):
        super().__init__()
        # Immediate features dictionary
        self.W_enc_imm = nn.Parameter(torch.zeros(d_in, d_sae//2))
        self.W_dec_imm = nn.Parameter(torch.zeros(d_sae//2, d_in))
        self.b_enc_imm = nn.Parameter(torch.zeros(d_sae//2))
        self.b_dec_imm = nn.Parameter(torch.zeros(d_in))

        # Contextual features dictionary (includes immediate features)
        self.W_enc_ctx = nn.Parameter(torch.zeros(d_in, d_sae))
        self.W_dec_ctx = nn.Parameter(torch.zeros(d_sae, d_in))
        self.b_enc_ctx = nn.Parameter(torch.zeros(d_sae))
        self.b_dec_ctx = nn.Parameter(torch.zeros(d_in))

        # Initialize weights
        for W in [self.W_enc_imm, self.W_dec_imm, self.W_enc_ctx, self.W_dec_ctx]:
            nn.init.kaiming_uniform_(W, nonlinearity='relu')
        for b in [self.b_enc_imm, self.b_dec_imm, self.b_enc_ctx, self.b_dec_ctx]:
            nn.init.uniform_(b, -0.01, 0.01)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.activation_dim = d_in
        self.dict_size = d_sae
        self.dtype = torch.float32
        self.context_size = context_size
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        
        if hook_name is None:
            hook_name = f"blocks.{hook_layer}.hook_resid_post"

        self.cfg = CustomSAEConfig(
            model_name=model_name,
            d_in=d_in,
            d_sae=d_sae,
            hook_name=hook_name,
            hook_layer=hook_layer,
            architecture="DualTemporal",
            activation_fn_str="relu",
            apply_b_dec_to_input=True,
            context_size=context_size
        )

    def encode_immediate(self, input_acts):
        pre_acts = (input_acts - self.b_dec_imm) @ self.W_enc_imm + self.b_enc_imm
        return torch.relu(pre_acts)

    def decode_immediate(self, acts):
        return (acts @ self.W_dec_imm) + self.b_dec_imm

    def encode_context(self, input_acts):
        # Input shape: (batch, context_size, d_in)
        pre_acts = (input_acts - self.b_dec_ctx.unsqueeze(1)) @ self.W_enc_ctx + self.b_enc_ctx
        return torch.relu(pre_acts)

    def decode_context(self, acts):
        return (acts @ self.W_dec_ctx) + self.b_dec_ctx

    def forward(self, acts, context=None, output_features=False):
        # Immediate features
        encoded_imm = self.encode_immediate(acts)
        decoded_imm = self.decode_immediate(encoded_imm)
        
        # Context features if context provided
        if context is not None:
            encoded_ctx = self.encode_context(context)
            decoded_ctx = self.decode_context(encoded_ctx)
            if output_features:
                return (decoded_imm, decoded_ctx), (encoded_imm, encoded_ctx)
            return decoded_imm, decoded_ctx
        
        if output_features:
            return decoded_imm, encoded_imm
        return decoded_imm

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

class DualTemporalTrainer(SAETrainer):
    def __init__(
        self,
        activation_dim=512,
        dict_size=64*512,
        lr=7e-5,
        warmup_steps=1000,
        context_size=8,
        lambda1=0.6,
        lambda2=0.4,
        alpha1=0.03,
        alpha2=0.06,
        steps=None,
        seed=None,
        device=None,
        layer=None,
        lm_name=None,
        wandb_name='DualTemporalTrainer',
        submodule_name=None
    ):
        super().__init__(seed)
        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        torch.manual_seed(seed or 42)
        self.device = device or 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.ae = DualTemporalSAE(
            d_in=activation_dim,
            d_sae=dict_size,
            hook_layer=layer,
            model_name=lm_name,
            context_size=context_size,
            lambda1=lambda1,
            lambda2=lambda2,
            alpha1=alpha1,
            alpha2=alpha2
        ).to(self.device)

        self.lr = lr
        self.warmup_steps = warmup_steps
        self.context_size = context_size
        self.context_buffer = []

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

        self.logging_parameters.extend(['immediate_loss', 'context_loss'])

    def loss(self, x, step, logging=False):
        x = x.to(self.ae.W_enc_imm.dtype)
        
        # Add current activation to context buffer
        self.context_buffer.append(x.detach())
        if len(self.context_buffer) > self.context_size:
            self.context_buffer.pop(0)
            
        # Immediate reconstruction loss
        encoded_imm = self.ae.encode_immediate(x)
        decoded_imm = self.ae.decode_immediate(encoded_imm)
        imm_loss = (x - decoded_imm).pow(2).sum(dim=-1).mean()
        
        # L1 regularization for immediate features
        l1_imm = self.ae.alpha1 * encoded_imm.abs().mean()
        
        total_loss = self.ae.lambda1 * (imm_loss + l1_imm)
        
        # Context loss if we have enough context
        ctx_loss = torch.tensor(0.0, device=self.device)
        l1_ctx = torch.tensor(0.0, device=self.device)
        
        if len(self.context_buffer) == self.context_size:
            context = torch.stack(self.context_buffer, dim=1)
            encoded_ctx = self.ae.encode_context(context)
            decoded_ctx = self.ae.decode_context(encoded_ctx)
            ctx_loss = (context - decoded_ctx).pow(2).sum(dim=-1).mean()
            l1_ctx = self.ae.alpha2 * encoded_ctx.abs().mean()
            total_loss += self.ae.lambda2 * (ctx_loss + l1_ctx)

        if logging:
            return {
                "loss": total_loss.item(),
                "immediate_loss": imm_loss.item(),
                "context_loss": ctx_loss.item(),
                "l1_immediate": l1_imm.item(),
                "l1_context": l1_ctx.item()
            }
        return total_loss

    def update(self, step, activations):
        activations = activations.to(self.device)
        self.optimizer.zero_grad()
        loss = self.loss(activations, step)
        loss.backward()

        # Normalize gradients for both immediate and context decoders
        with torch.no_grad():
            for W_dec in [self.ae.W_dec_imm, self.ae.W_dec_ctx]:
                if W_dec.grad is not None:
                    W_dec.grad = remove_gradient_parallel_to_decoder_directions(
                        W_dec.T,
                        W_dec.grad.T,
                        self.ae.activation_dim,
                        W_dec.shape[0]
                    ).T

        torch.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
        self.optimizer.step()
        self.lr_scheduler.step()

        # Normalize decoder weights
        with torch.no_grad():
            self.ae.W_dec_imm.data = set_decoder_norm_to_unit_norm(
                self.ae.W_dec_imm.T,
                self.ae.activation_dim,
                self.ae.W_dec_imm.shape[0]
            ).T
            self.ae.W_dec_ctx.data = set_decoder_norm_to_unit_norm(
                self.ae.W_dec_ctx.T,
                self.ae.activation_dim,
                self.ae.W_dec_ctx.shape[0]
            ).T

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
    jump_coeff: float = 0.1,
    context_size: int = 8,
    lambda1: float = 0.6,
    lambda2: float = 0.4,
    alpha1: float = 0.03,
    alpha2: float = 0.06
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
    trainer = DualTemporalTrainer(
        activation_dim=activation_dim,
        dict_size=dict_size,
        lr=learning_rate,
        warmup_steps=warmup_steps,
        context_size=8,
        lambda1=0.6,
        lambda2=0.4, 
        alpha1=0.03,
        alpha2=0.06,
        steps=steps,
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
            warmup_steps=1000,
            seed=42,
            wandb_logging=not no_wandb_logging,
            wandb_entity=None,
            wandb_project=None,
            context_size=8,
            lambda1=0.6,
            lambda2=0.4,
            alpha1=0.03,
            alpha2=0.06
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
