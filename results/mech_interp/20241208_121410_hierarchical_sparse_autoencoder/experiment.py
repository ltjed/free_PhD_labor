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

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, 
                 l1_coef_layer1: float = 0.1, l1_coef_layer2: float = 0.2,
                 consistency_coef: float = 0.1):
        super().__init__()
        # First layer
        self.encoder1 = nn.Linear(input_dim, hidden_dim)
        self.gate1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Second layer
        self.encoder2 = nn.Linear(hidden_dim, latent_dim)
        self.gate2 = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.Sigmoid()
        )
        
        # Decoders
        self.decoder2 = nn.Linear(latent_dim, hidden_dim)
        self.decoder1 = nn.Linear(hidden_dim, input_dim)
        
        # Loss coefficients
        self.l1_coef_layer1 = l1_coef_layer1
        self.l1_coef_layer2 = l1_coef_layer2
        self.consistency_coef = consistency_coef
        
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # First layer
        h1 = self.encoder1(x)
        g1 = self.gate1(x)
        h1_gated = h1 * g1
        h1_activated = torch.relu(h1_gated)
        
        # Second layer
        h2 = self.encoder2(h1_activated)
        g2 = self.gate2(h1_activated)
        h2_gated = h2 * g2
        h2_activated = torch.relu(h2_gated)
        
        # Decoding
        h1_reconstructed = self.decoder2(h2_activated)
        x_reconstructed = self.decoder1(h1_reconstructed)
        
        return x_reconstructed, h1_activated, h2_activated, g1, g2

    def get_loss(self, x: Tensor) -> Tuple[Tensor, Dict]:
        x_reconstructed, h1, h2, g1, g2 = self(x)
        
        # Reconstruction loss
        mse_loss = nn.MSELoss()(x_reconstructed, x)
        
        # L1 sparsity losses
        l1_loss_layer1 = self.l1_coef_layer1 * torch.mean(torch.abs(h1))
        l1_loss_layer2 = self.l1_coef_layer2 * torch.mean(torch.abs(h2))
        
        # Inter-layer consistency loss
        consistency_loss = self.consistency_coef * torch.mean(
            torch.abs(torch.cosine_similarity(h1.unsqueeze(2), h2.unsqueeze(1), dim=0))
        )
        
        # Total loss
        total_loss = mse_loss + l1_loss_layer1 + l1_loss_layer2 + consistency_loss
        
        metrics = {
            "mse_loss": mse_loss.item(),
            "l1_loss_layer1": l1_loss_layer1.item(),
            "l1_loss_layer2": l1_loss_layer2.item(),
            "consistency_loss": consistency_loss.item(),
            "sparsity_layer1": torch.mean((h1 > 0).float()).item(),
            "sparsity_layer2": torch.mean((h2 > 0).float()).item(),
            "gate1_mean": torch.mean(g1).item(),
            "gate2_mean": torch.mean(g2).item()
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
    model: SparseAutoencoder,
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
    autoencoder: SparseAutoencoder,
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
    
    # Get both layer activations
    _, h1, h2, g1, g2 = autoencoder(activations.view(-1, activations.size(-1)))
    
    metrics = {
        "neuron_sparsity": torch.mean((activations > 0).float()).item(),
        "layer1_sparsity": torch.mean((h1 > 0).float()).item(),
        "layer2_sparsity": torch.mean((h2 > 0).float()).item(),
        "activation_entropy": compute_entropy(activations),
        "layer1_entropy": compute_entropy(h1),
        "layer2_entropy": compute_entropy(h2),
        "gate1_mean": torch.mean(g1).item(),
        "gate2_mean": torch.mean(g2).item(),
        "layer1_feature_coherence": compute_feature_coherence(h1),
        "layer2_feature_coherence": compute_feature_coherence(h2),
        "cross_layer_similarity": torch.mean(torch.cosine_similarity(
            h1.unsqueeze(2), h2.unsqueeze(1), dim=0)).item()
    }
    
    return metrics

def compute_feature_coherence(activations: Tensor) -> float:
    # Compute pairwise cosine similarities between feature vectors
    normalized = torch.nn.functional.normalize(activations, dim=0)
    similarities = torch.mm(normalized.T, normalized)
    
    # Average similarity excluding self-similarity
    mask = ~torch.eye(similarities.shape[0], dtype=bool, device=similarities.device)
    coherence = torch.mean(similarities[mask]).item()
    return coherence

def compute_entropy(x: Tensor) -> float:
    # Compute activation distribution entropy
    probs = torch.histc(x.float(), bins=50) / x.numel()
    probs = probs[probs > 0]  # Remove zero probabilities
    return -torch.sum(probs * torch.log(probs)).item()

def run(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    
    # Train autoencoder on intermediate layer activations
    layer_idx = model_config.n_layer // 2  # Use middle layer
    input_dim = model_config.n_embd
    latent_dim = input_dim // 2
    
    hidden_dim = input_dim // 2
    latent_dim = hidden_dim // 2
    autoencoder = SparseAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        l1_coef_layer1=0.1,
        l1_coef_layer2=0.2,
        consistency_coef=0.1
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
    
    # Evaluate interpretability
    metrics = evaluate_interpretability(
        model, autoencoder, val_loader, layer_idx, device
    )
    
    # Save results
    results = {
        "transformer_training": transformer_log,
        "autoencoder_training": autoencoder_log,
        "interpretability_metrics": metrics
    }
    
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Save models
    torch.save(model.state_dict(), os.path.join(out_dir, "transformer.pt"))
    torch.save(autoencoder.state_dict(), os.path.join(out_dir, "autoencoder.pt"))
    
    wandb.finish()
