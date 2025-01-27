import torch
import os
import torch.nn as nn
import torch.nn.functional as F
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
    """Convert HuggingFace dataset to generator."""
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
        n_ctxs=2048,
        ctx_len=128,
        refresh_batch_size=24,
        out_batch_size=2048,
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
        
        # Initialize buffer state
        self.buffer = []
        self.buffer_idx = 0
        self.is_exhausted = False
        self.current_batch = None
        
    def __iter__(self):
        self.buffer = []
        self.buffer_idx = 0
        return self
        
    def __next__(self):
        """Get next batch of activations."""
        try:
            # Check if iterator is exhausted
            if self.is_exhausted:
                raise StopIteration
                
            # Initialize or refresh buffer if needed
            if self.buffer_idx >= len(self.buffer):
                try:
                    self._refresh_buffer()
                except StopIteration:
                    self.is_exhausted = True
                    raise
                self.buffer_idx = 0
            
            # Prepare batch
            remaining_size = self.out_batch_size
            batch = []
            
            # Fill batch from current buffer
            while remaining_size > 0:
                if self.buffer_idx >= len(self.buffer):
                    try:
                        self._refresh_buffer()
                        self.buffer_idx = 0
                    except StopIteration:
                        break
                
                # Calculate how many items we can take from current buffer
                items_to_take = min(remaining_size, len(self.buffer) - self.buffer_idx)
                batch.extend(self.buffer[self.buffer_idx:self.buffer_idx + items_to_take])
                self.buffer_idx += items_to_take
                remaining_size -= items_to_take
            
            # If we couldn't fill the batch at all
            if not batch:
                self.is_exhausted = True
                raise StopIteration
            
            # Stack and return batch
            self.current_batch = torch.stack(batch).detach()
            return self.current_batch
            
        except Exception as e:
            print(f"Error in __next__: {str(e)}")
            if not isinstance(e, StopIteration):
                print(f"Buffer state: size={len(self.buffer)}, idx={self.buffer_idx}")
            raise StopIteration
            
    def _refresh_buffer(self):
        """Refresh internal buffer with new activations."""
        self.buffer = []
        texts = []
        
        # Collect texts
        for _ in range(self.refresh_batch_size):
            try:
                text = next(self.generator)
                if not isinstance(text, str) or not text.strip():
                    print("Warning: Invalid text input")
                    continue
                texts.append(text)
            except StopIteration:
                if not texts:  # If we couldn't get any texts
                    raise
                break
        
        # Process texts
        try:
            tokens = self.model.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.ctx_len,
                return_tensors="pt"
            ).to(self.device)
            
            # Capture activations
            with torch.no_grad():
                with self.model.invoke(tokens) as invoker:
                    def capture_activations(acts):
                        if isinstance(acts, (list, tuple)):
                            self.buffer.extend([a.detach() for a in acts])
                        else:
                            self.buffer.append(acts.detach())
                    
                    invoker.register_hook(self.submodule, capture_activations)
                    outputs = invoker.forward()
            
            if not self.buffer:
                print("Warning: No activations captured in buffer")
                print(f"Input texts: {len(texts)}, Tokens shape: {tokens.input_ids.shape}")
                raise StopIteration
                
        except Exception as e:
            print(f"Error refreshing buffer: {str(e)}")
            raise StopIteration
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
    """Implementation of SAE with top-k orthogonality constraints."""
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        hook_layer: int,
        model_name: str = "EleutherAI/pythia-70m-deduped",
        hook_name: Optional[str] = None,
        topk_percent: float = 0.001,  # 0.1%
        tau: float = 0.1,
    ):
        super().__init__()
        self.W_enc = nn.Parameter(torch.zeros(d_in, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        
        # Orthogonality parameters
        self.topk_percent = topk_percent
        self.tau = tau
        self.tau_history = []  # Track tau values
        self.correlation_history = []  # Track correlation distributions
        
        # Register L2 normalization hook
        self._register_norm_hook()
        
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

    def _register_norm_hook(self):
        def normalize_decoder(module, grad_input, grad_output):
            with torch.no_grad():
                self.W_dec.data = F.normalize(self.W_dec.data, dim=0)
        self.register_backward_hook(normalize_decoder)
        
    def encode(self, input_acts):
        pre_acts = (input_acts - self.b_dec) @ self.W_enc + self.b_enc
        acts = torch.relu(pre_acts)
        return acts
        
    def get_top_pairs(self, encoded):
        """Select top-k most correlated feature pairs."""
        B = encoded.size(0)
        
        # Compute pairwise correlations
        norm_encoded = F.normalize(encoded, dim=0)
        correlations = torch.mm(norm_encoded.t(), norm_encoded)
        
        # Update adaptive tau based on correlation distribution
        with torch.no_grad():
            corr_vals = correlations[~torch.eye(correlations.size(0), device=correlations.device).bool()]
            mean_corr = corr_vals.mean()
            std_corr = corr_vals.std()
            self.tau = mean_corr + std_corr  # Adaptive threshold
            
            # Store history
            self.tau_history.append(self.tau.item())
            self.correlation_history.append((mean_corr.item(), std_corr.item()))
        
        # Get top k% pairs excluding diagonal
        mask = torch.eye(correlations.size(0), device=correlations.device).bool()
        correlations.masked_fill_(mask, float('-inf'))
        
        k = int(self.topk_percent * correlations.numel())
        top_corr, indices = torch.topk(correlations.view(-1), k)
        
        row_idx = indices // correlations.size(1)
        col_idx = indices % correlations.size(1)
        
        return row_idx, col_idx, top_corr

    def decode(self, acts):
        return (acts @ self.W_dec) + self.b_dec

    def forward(self, acts, output_features=False):
        encoded = self.encode(acts)
        decoded = self.decode(encoded)
        
        # Store pairs for loss computation
        self.current_pairs = self.get_top_pairs(encoded)
        
        if output_features:
            return decoded, encoded
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
class TopKOrthogonalTrainer(SAETrainer):
    """Trainer for SAE with top-k orthogonality constraints."""
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
        self.ae = TopKOrthogonalSAE(d_in=activation_dim, d_sae=dict_size, hook_layer=layer, model_name=lm_name)

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
        
        # Reconstruction and sparsity losses
        l2_loss = torch.linalg.norm(x - x_hat, dim=-1).mean()
        l1_loss = f.norm(p=1, dim=-1).mean()
        
        # Orthogonality loss on top pairs
        row_idx, col_idx, correlations = self.ae.current_pairs
        ortho_loss = torch.mean(F.relu(correlations - self.ae.tau))

        if self.steps_since_active is not None:
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
    current_step = 0
    
    # Training loop with validation
    successful_steps = 0
    for step in range(steps):
        try:
            # Get and validate activations
            activations = next(activation_buffer)
            if activations is None or activations.nelement() == 0:
                print(f"Warning: Empty activations at step {step}")
                continue
                
            print(f"Batch {step}: activation shape = {activations.shape}, norm = {activations.norm()}")
            
            # Compute loss and update
            loss_info = trainer.loss(activations, logging=True)
            if loss_info is None:
                print(f"Warning: Invalid loss computation at step {step}")
                continue
                
            trainer.update(step, activations)
            training_log.append(loss_info.losses)
            successful_steps += 1
            
            if step % 100 == 0:
                print(f"Step {step} (successful: {successful_steps}): {loss_info.losses}")
                print(f"Current tau: {trainer.ae.tau_history[-1] if trainer.ae.tau_history else 'N/A'}")
                print(f"Correlation stats: {trainer.ae.correlation_history[-1] if trainer.ae.correlation_history else 'N/A'}")
        except StopIteration:
            print("Data iterator exhausted, breaking training loop")
            break
        except Exception as e:
            print(f"Error in training loop: {str(e)}")
            raise
            
            if wandb_logging and wandb_entity and wandb_project:
                import wandb
                wandb.log(loss_info.losses, step=step)

    # Prepare final results
    final_info = {
        "training_steps": len(training_log),
        "final_loss": training_log[-1]["loss"] if training_log else None,
        "layer": layer,
        "dict_size": dict_size,
        "learning_rate": learning_rate,
        "sparsity_penalty": sparsity_penalty,
        "tau_history": trainer.ae.tau_history,
        "correlation_history": trainer.ae.correlation_history
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
        "final_info": final_info,
        "tau_history": trainer.ae.tau_history,
        "correlation_history": trainer.ae.correlation_history
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



    # Save trained models
    for k in range(len(layers)):
        sae = saes[k].to(dtype=str_to_dtype(llm_dtype))
        sae.cfg.dtype = llm_dtype
        
        # Save model checkpoint
        checkpoint = {
            "model_state_dict": sae.state_dict(),
            "config": sae.cfg,
            "layer": layers[k]
        }
        checkpoint_path = os.path.join(save_dir, f"sae_layer_{layers[k]}.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
