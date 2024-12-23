import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import numpy as np
import os
from typing import Dict, List, Tuple
import wandb
import json
import argparse
from contextlib import nullcontext

# collect datasets
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
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, 
                 bottom_l1_coef: float = 0.9, top_l1_coef: float = 0.7):
        super().__init__()
        # Bottom layer (90% sparsity target)
        self.encoder1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder1 = nn.Linear(hidden_dim, input_dim)
        
        # Top layer (70% sparsity target)
        self.encoder2 = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        self.decoder2 = nn.Linear(latent_dim, hidden_dim)
        
        # Skip connection
        self.skip_proj = nn.Linear(input_dim, hidden_dim)
        
        self.bottom_l1_coef = bottom_l1_coef
        self.top_l1_coef = top_l1_coef
        
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # Bottom layer
        skip = self.skip_proj(x)
        h1 = self.encoder1(x)
        
        # Top layer
        h2 = self.encoder2(h1)
        
        # Decoding with skip connection
        decoded_h1 = self.decoder2(h2)
        decoded_x = self.decoder1(decoded_h1 + skip)
        
        return decoded_x, h1, h2

    def get_loss(self, x: Tensor, training_phase: int = 3) -> Tuple[Tensor, Dict]:
        decoded_x, h1, h2 = self(x)
        
        # MSE reconstruction loss
        mse_loss = nn.MSELoss()(decoded_x, x)
        
        # L1 penalties for sparsity
        l1_bottom = self.bottom_l1_coef * torch.mean(torch.abs(h1))
        l1_top = self.top_l1_coef * torch.mean(torch.abs(h2))
        
        # Phase-specific training
        if training_phase == 1:
            # Train only bottom layer
            total_loss = mse_loss + l1_bottom
        elif training_phase == 2:
            # Train only top layer
            total_loss = mse_loss + l1_top
        else:
            # End-to-end fine-tuning
            total_loss = mse_loss + l1_bottom + l1_top
            
        metrics = {
            "mse_loss": mse_loss.item(),
            "l1_bottom": l1_bottom.item(),
            "l1_top": l1_top.item(),
            "bottom_sparsity": torch.mean((h1 > 0).float()).item(),
            "top_sparsity": torch.mean((h2 > 0).float()).item()
        }
        
        return total_loss, metrics

def train_transformer(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int
) -> List[Dict]:
    model.train()
    training_log = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in train_loader:
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
            
        metrics = {"epoch": epoch, "loss": epoch_loss / len(train_loader)}
        training_log.append(metrics)
        wandb.log(metrics)
        
    return training_log

def train_autoencoder(
    model: HierarchicalSparseAutoencoder,
    activations: Tensor,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    phase: int = 3
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
        "latent_sparsity": torch.mean((latents > 0).float()).item(),
        "activation_entropy": compute_entropy(activations),
        "latent_entropy": compute_entropy(latents)
    }
    
    return metrics

def compute_entropy(x: Tensor) -> float:
    # Compute activation distribution entropy
    probs = torch.histc(x.float(), bins=50) / x.numel()
    probs = probs[probs > 0]  # Remove zero probabilities
    return -torch.sum(probs * torch.log(probs)).item()

def run(out_dir: str):
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
    
    train_dataset = OpenWebTextDataset("train", max_length=128)
    val_dataset = OpenWebTextDataset("validation", max_length=128)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Train transformer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    transformer_log = train_transformer(model, train_loader, optimizer, device, num_epochs=10)

    # Save transformer checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(out_dir, 'transformer_checkpoint.pt'))
    
    # Train autoencoder on intermediate layer activations
    layer_idx = model_config.n_layer // 2  # Use middle layer
    input_dim = model_config.n_embd
    latent_dim = input_dim // 2
    
    hidden_dim = input_dim // 2
    latent_dim = hidden_dim // 2
    
    autoencoder = HierarchicalSparseAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim
    ).to(device)
    
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

    # Phase 1: Train bottom layer
    optimizer = torch.optim.Adam(
        list(autoencoder.encoder1.parameters()) + 
        list(autoencoder.decoder1.parameters()) +
        list(autoencoder.skip_proj.parameters()),
        lr=1e-3
    )
    phase1_log = train_autoencoder(
        autoencoder, activations, optimizer, num_epochs=10, device=device, phase=1
    )
    
    # Phase 2: Train top layer
    optimizer = torch.optim.Adam(
        list(autoencoder.encoder2.parameters()) + 
        list(autoencoder.decoder2.parameters()),
        lr=1e-3
    )
    phase2_log = train_autoencoder(
        autoencoder, activations, optimizer, num_epochs=10, device=device, phase=2
    )
    
    # Phase 3: End-to-end fine-tuning
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-4)
    
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
        "phase1_loss": phase1_log[-1]["total_loss"],
        "phase2_loss": phase2_log[-1]["total_loss"],
        "final_loss": autoencoder_log[-1]["total_loss"],
        "bottom_sparsity": metrics["bottom_sparsity"],
        "top_sparsity": metrics["top_sparsity"]
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
    print("Started running experiment.py")
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Run mech interpretation experiment")
    parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
    args = parser.parse_args()
    
    # Run experiment
    run(args.out_dir)
