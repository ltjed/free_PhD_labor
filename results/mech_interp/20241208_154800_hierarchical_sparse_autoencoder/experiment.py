import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import time
from tqdm import tqdm
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import numpy as np
import os
from typing import Dict, List, Tuple
import wandb
import json
import argparse
from contextlib import nullcontext

class OpenWebTextDataset(Dataset):
    def __init__(self, split: str, max_length: int = 128):
        self.dataset = load_dataset("openwebtext", split=split)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        text = self.dataset[idx]["text"]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze()
        }

class HierarchicalSparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, primary_dim: int, secondary_dim: int, 
                 primary_l1: float = 0.1, secondary_l1: float = 0.2):
        super().__init__()
        
        # Primary level for high-level concepts
        self.primary_encoder = nn.Sequential(
            nn.Linear(input_dim, primary_dim),
            nn.ReLU()
        )
        self.primary_decoder = nn.Linear(primary_dim, input_dim)
        
        # Secondary level for detailed features
        self.secondary_encoder = nn.Sequential(
            nn.Linear(input_dim, secondary_dim),
            nn.ReLU()
        )
        self.secondary_decoder = nn.Linear(secondary_dim, input_dim)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(primary_dim + secondary_dim, primary_dim + secondary_dim),
            nn.Softmax(dim=-1)
        )
        
        self.primary_l1 = primary_l1
        self.secondary_l1 = secondary_l1
        
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # Encode both levels
        primary_latents = self.primary_encoder(x)
        secondary_latents = self.secondary_encoder(x)
        
        # Compute attention weights
        combined_latents = torch.cat([primary_latents, secondary_latents], dim=-1)
        attention_weights = self.attention(combined_latents)
        
        # Split attention weights
        primary_attention = attention_weights[..., :primary_latents.size(-1)]
        secondary_attention = attention_weights[..., primary_latents.size(-1):]
        
        # Apply attention
        attended_primary = primary_latents * primary_attention
        attended_secondary = secondary_latents * secondary_attention
        
        # Decode both levels
        primary_reconstruction = self.primary_decoder(attended_primary)
        secondary_reconstruction = self.secondary_decoder(attended_secondary)
        
        # Combined reconstruction
        reconstructed = (primary_reconstruction + secondary_reconstruction) / 2
        
        return reconstructed, primary_latents, secondary_latents, attention_weights

    def get_loss(self, x: Tensor) -> Tuple[Tensor, Dict]:
        reconstructed, primary_latents, secondary_latents, attention_weights = self(x)
        
        # Reconstruction loss
        mse_loss = nn.MSELoss()(reconstructed, x)
        
        # Level-specific sparsity constraints
        primary_l1_loss = self.primary_l1 * torch.mean(torch.abs(primary_latents))
        secondary_l1_loss = self.secondary_l1 * torch.mean(torch.abs(secondary_latents))
        
        # Total loss
        total_loss = mse_loss + primary_l1_loss + secondary_l1_loss
        
        metrics = {
            "mse_loss": mse_loss.item(),
            "primary_l1_loss": primary_l1_loss.item(),
            "secondary_l1_loss": secondary_l1_loss.item(),
            "primary_sparsity": torch.mean((primary_latents > 0).float()).item(),
            "secondary_sparsity": torch.mean((secondary_latents > 0).float()).item(),
            "attention_entropy": self._compute_attention_entropy(attention_weights).item()
        }
        
        return total_loss, metrics
        
    def _compute_attention_entropy(self, attention_weights: Tensor) -> Tensor:
        # Compute entropy of attention distribution
        entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-10), dim=-1)
        return torch.mean(entropy)

def train_transformer(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
) -> List[Dict]:
    model.train()
    training_log = []
    epoch_loss = 0.0
    
    with tqdm(train_loader, desc=f"Epoch {epoch}") as pbar:
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            
            loss = outputs.loss
            epoch_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Update progress bar
        pbar.set_postfix({"loss": loss.item()})
            
        # Log every 100 batches
        if len(training_log) % 100 == 0:
            metrics = {
                "epoch": epoch,
                "batch_loss": loss.item(),
                "avg_loss": epoch_loss / (len(training_log) + 1)
            }
            training_log.append(metrics)
            wandb.log(metrics)
        
    return training_log

def train_autoencoder(
    model: HierarchicalSparseAutoencoder,
    activations: Tensor,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device
) -> List[Dict]:
    model.train()
    training_log = []
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        metrics_sum = {"mse_loss": 0.0, "l1_loss": 0.0, "sparsity": 0.0}
        
        for i in range(0, len(activations), 128):
            batch = activations[i:i + 128].to(device)
            loss, batch_metrics = model.get_loss(batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            for k, v in batch_metrics.items():
                metrics_sum[k] += v
                
        avg_metrics = {k: v / (len(activations) // 128) for k, v in metrics_sum.items()}
        metrics = {"epoch": epoch, "total_loss": total_loss, **avg_metrics}
        training_log.append(metrics)
        wandb.log(metrics)
        
    return training_log

def evaluate_interpretability(
    transformer: nn.Module,
    autoencoder: HierarchicalSparseAutoencoder,
    val_loader: DataLoader,
    layer_idx: int,
    device: torch.device
) -> Dict:
    transformer.eval()
    autoencoder.eval()
    
    # Collect activations and decoded features
    all_activations = []
    all_latents = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Get intermediate activations
            outputs = transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            activations = outputs.hidden_states[layer_idx]
            
            # Get SAE latents
            _, latents = autoencoder(activations.view(-1, activations.size(-1)))
            
            all_activations.append(activations.cpu())
            all_latents.append(latents.cpu())
    
    # Compute interpretability metrics
    activations = torch.cat(all_activations, dim=0)
    latents = torch.cat(all_latents, dim=0)
    
    metrics = {
        "neuron_sparsity": torch.mean((activations > 0).float()).item(),
        "primary_latent_sparsity": torch.mean((latents[0] > 0).float()).item(),
        "secondary_latent_sparsity": torch.mean((latents[1] > 0).float()).item(),
        "activation_entropy": compute_entropy(activations),
        "primary_latent_entropy": compute_entropy(latents[0]),
        "secondary_latent_entropy": compute_entropy(latents[1]),
        "level_correlation": compute_level_correlation(latents[0], latents[1]).item()
    }
    
    return metrics

def compute_level_correlation(primary: Tensor, secondary: Tensor) -> Tensor:
    # Compute correlation between primary and secondary level activations
    primary_flat = primary.view(-1, primary.size(-1))
    secondary_flat = secondary.view(-1, secondary.size(-1))
    
    # Compute correlation matrix
    corr_matrix = torch.corrcoef(torch.cat([primary_flat, secondary_flat], dim=0))
    
    # Extract cross-correlation block
    n_primary = primary_flat.size(1)
    cross_corr = corr_matrix[:n_primary, n_primary:]
    
    # Return mean absolute correlation
    return torch.mean(torch.abs(cross_corr))

def compute_entropy(x: Tensor) -> float:
    # Compute activation distribution entropy
    probs = torch.histc(x.float(), bins=50) / x.numel()
    probs = probs[probs > 0]  # Remove zero probabilities
    return -torch.sum(probs * torch.log(probs)).item()

def run(out_dir: str, timeout_minutes: int = 120):
    # Set up output directory and device
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type="cuda")
    
    # Initialize wandb
    wandb.init(project="mech-interp-experiment", dir=out_dir)
    
    # Load model and dataset
    model_config = GPT2Config(
        n_layer=6,
        n_head=8,
        n_embd=512,
        vocab_size=50257
    )
    model = GPT2LMHeadModel(model_config).to(device)
    model.gradient_checkpointing_enable()  # Enable gradient checkpointing
    
    train_dataset = OpenWebTextDataset("train", max_length=128)
    val_dataset = OpenWebTextDataset("validation", max_length=128)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Train transformer with timeout
    start_time = time.time()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    transformer_log = []
    
    for epoch in range(10):
        if (time.time() - start_time) > (timeout_minutes * 60):
            print(f"Timeout reached after {epoch} epochs")
            break
            
        epoch_metrics = train_transformer(
            model, train_loader, optimizer, device, epoch
        )
        transformer_log.extend(epoch_metrics)
        
        # Early stopping check
        if epoch > 2 and transformer_log[-1]["loss"] > transformer_log[-2]["loss"] * 0.99:
            print("Early stopping triggered")
            break

    # Save transformer checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(out_dir, 'transformer_checkpoint.pt'))
    
    # Train autoencoder on intermediate layer activations
    layer_idx = model_config.n_layer // 2  # Use middle layer
    input_dim = model_config.n_embd
    latent_dim = input_dim // 2
    
    # Initialize hierarchical autoencoder
    primary_dim = input_dim // 2
    secondary_dim = input_dim // 4
    autoencoder = HierarchicalSparseAutoencoder(
        input_dim=input_dim,
        primary_dim=primary_dim,
        secondary_dim=secondary_dim
    ).to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
    
    # Collect activations for autoencoder training
    activations = []
    model.eval()
    with torch.no_grad():
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            layer_activations = outputs.hidden_states[layer_idx]
            activations.append(layer_activations.cpu())
    
    activations = torch.cat(activations, dim=0).view(-1, input_dim)
    autoencoder_log = train_autoencoder(
        autoencoder, activations, optimizer, num_epochs=20, device=device
    )
    
    # Save autoencoder checkpoint
    torch.save({
        'model_state_dict': autoencoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(out_dir, 'autoencoder_checkpoint.pt'))
    
    # Evaluate interpretability
    metrics = evaluate_interpretability(
        model, autoencoder, val_loader, layer_idx, device
    )
    
    # Prepare final results
    final_info = {
        "transformer_loss": transformer_log[-1]["loss"],
        "autoencoder_loss": autoencoder_log[-1]["total_loss"],
        "reconstruction_error": metrics["reconstruction_error"],
        "sparsity": metrics["sparsity"]
    }
    
    # Save detailed results
    results = {
        "transformer_training": transformer_log,
        "autoencoder_training": autoencoder_log,
        "interpretability_metrics": metrics
    }
    
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
        
    # Save final info in required format
    final_info_formatted = {
        metric_name: {"means": value}
        for metric_name, value in final_info.items()
    }
    
    with open(os.path.join(out_dir, "final_info.json"), "w") as f:
        json.dump(final_info_formatted, f)
        
    wandb.finish()
    return final_info

if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Run mechanical interpretation experiment")
    parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
    args = parser.parse_args()
    
    # Run experiment
    run(args.out_dir)
