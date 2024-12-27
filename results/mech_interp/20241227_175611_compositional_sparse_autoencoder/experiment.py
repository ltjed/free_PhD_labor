import os
import json
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from torch.nn import CrossEntropyLoss
from contextlib import nullcontext
from typing import Dict, List
import numpy as np

from datasets import load_dataset  # Make sure you have: pip install datasets fsspec pyarrow

# -------------------------------
# 1) Define Model & Autoencoder
# -------------------------------
class CompositionalSparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, feature_dim: int = 256, composition_dim: int = 128, 
                 n_heads: int = 8, sparsity_k: int = 32):
        super().__init__()
        self.feature_dim = feature_dim
        self.composition_dim = composition_dim
        self.sparsity_k = sparsity_k
        
        # Feature extractor stream
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU()
        )
        
        # Composition network with multi-head attention
        self.composition_attention = nn.MultiheadAttention(
            embed_dim=composition_dim,
            num_heads=n_heads,
            batch_first=True
        )
        
        # Projection layers
        self.feature_to_composition = nn.Linear(feature_dim, composition_dim)
        self.composition_to_feature = nn.Linear(composition_dim, feature_dim)
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )
        
        # Bottleneck layer
        self.bottleneck = nn.Linear(feature_dim, input_dim)
        
    def get_top_k_mask(self, x):
        """Keep only top-k activations per sample."""
        topk = torch.topk(x, k=self.sparsity_k, dim=-1)
        mask = torch.zeros_like(x)
        mask.scatter_(-1, topk.indices, 1.0)
        return mask
        
    def forward(self, x):
        # Feature extraction
        features = self.feature_encoder(x)
        
        # Apply sparsity
        features = features * self.get_top_k_mask(features)
        
        # Composition network
        comp_input = self.feature_to_composition(features).unsqueeze(1)
        comp_output, attention_weights = self.composition_attention(
            comp_input, comp_input, comp_input
        )
        comp_output = comp_output.squeeze(1)
        
        # Project back to feature space
        composed_features = self.composition_to_feature(comp_output)
        
        # Residual connection with gating
        gate_values = self.gate(torch.cat([features, composed_features], dim=-1))
        gated_features = gate_values * composed_features + (1 - gate_values) * features
        
        # Final reconstruction
        reconstruction = self.bottleneck(gated_features)
        
        return reconstruction, features, attention_weights

class ModifiedGPT2(GPT2LMHeadModel):
    """
    GPT-2 model that can optionally accept a 'custom_hidden_states' dict
    to replace the hidden states at certain layers.
    """
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        custom_hidden_states=None,
        output_hidden_states=False,
    ):
        if custom_hidden_states is None:
            # Normal forward pass
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=output_hidden_states,
            )
        
        # If custom_hidden_states is provided
        inputs_embeds = self.transformer.wte(input_ids)
        position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
        position_embeds = self.transformer.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        
        all_hidden_states = () if not output_hidden_states else (hidden_states,)
        
        for i, block in enumerate(self.transformer.h):
            if i in custom_hidden_states:
                hidden_states = custom_hidden_states[i]
            else:
                layer_outputs = block(hidden_states, attention_mask=attention_mask)
                hidden_states = layer_outputs[0]
            
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

        hidden_states = self.transformer.ln_f(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return {
            'loss': loss,
            'logits': lm_logits,
            'hidden_states': all_hidden_states if output_hidden_states else None
        }

# -------------------------------
# 2) Define Dataset Class
# -------------------------------
class OpenWebTextSubset(Dataset):
    """
    Loads a tiny subset from the openwebtext dataset for quick testing.

    We do:
      - full_data = load_dataset("openwebtext", split="train")
      - sub_data_train = full_data.select(range(0, N)) for training
      - sub_data_val   = full_data.select(range(N, 2N)) for validation
    """
    def __init__(self, indices_range, max_length: int = 128):
        # Load the entire "train" split (openwebtext only has 'train')
        dataset_full = load_dataset("openwebtext", split="train")
        # Select a small subset
        dataset_small = dataset_full.select(indices_range)

        self.samples = dataset_small
        self.max_length = max_length

        # GPT2 tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        # GPT2 has no pad_token by default, so set it to eos_token
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]["text"]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

# -------------------------------
# 3) Compute Functions
# -------------------------------
def compute_entropy(tensor: torch.Tensor) -> float:
    """
    Compute an approximate empirical entropy of the last dimension of a tensor by:
    1. Flattening all but the feature dimension
    2. Computing a histogram
    3. Summation of -p * log(p)
    """
    flat_tensor = tensor.view(-1, tensor.size(-1))
    hist = torch.histc(flat_tensor, bins=100)
    probs = hist / hist.sum()
    probs = probs[probs > 0]
    return -(probs * torch.log(probs)).sum().item()

def train_transformer(model, train_loader, optimizer, device, num_epochs=1000):
    model.train()
    log = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device).float()  # [batch, seq_len]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )

            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        log.append({"epoch": epoch, "loss": avg_loss})
        print(f"[Transformer] Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return log

def train_autoencoder(autoencoder, activations, optimizer, device, num_epochs=2):
    autoencoder.train()
    log = []

    # Ensure activations are on the correct device
    activations = activations.to(device)

    for epoch in range(num_epochs):
        reconstruction, features, attention_weights = autoencoder(activations)

        # Reconstruction loss
        recon_loss = F.mse_loss(reconstruction, activations)
        
        # Feature diversity loss
        feature_correlation = torch.matmul(features.T, features)
        diversity_loss = torch.norm(
            feature_correlation - torch.eye(feature_correlation.shape[0], device=device)
        )
        
        # Attention entropy regularization
        attention_entropy = -torch.mean(
            torch.sum(attention_weights * torch.log(attention_weights + 1e-10), dim=-1)
        )
        
        total_loss = recon_loss + 0.1 * diversity_loss + 0.01 * attention_entropy

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        log.append({
            "epoch": epoch,
            "total_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "diversity_loss": diversity_loss.item(),
            "attention_entropy": attention_entropy.item()
        })
        
        if epoch % 100 == 0:
            print(
                f"[Autoencoder] Epoch {epoch+1}/{num_epochs}, "
                f"Total Loss: {total_loss.item():.4f}, "
                f"Recon Loss: {recon_loss.item():.4f}"
            )

    return log

def evaluate_interpretability(
    transformer: nn.Module,
    autoencoder: CompositionalSparseAutoencoder,
    val_loader: DataLoader,
    layer_idx: int,
    device: torch.device
):
    transformer.eval()
    autoencoder.eval()

    original_losses = []
    reconstructed_losses = []
    all_activations = []
    all_latents = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device).float()
            # Manually reshape the attention mask so it can broadcast
            # to [batch, n_heads, seq_len, seq_len] internally.
            # We do [batch, 1, 1, seq_len].
            bsz, seq_len = attention_mask.shape
            attention_mask = attention_mask.view(bsz, 1, 1, seq_len)

            # 1) Original model pass
            outputs = transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
                output_hidden_states=True
            )
            original_loss = outputs["loss"]
            activations = outputs["hidden_states"][layer_idx]  # [bsz, seq_len, embed_dim]

            # 2) Autoencoder pass
            flat_activations = activations.view(-1, activations.size(-1))
            reconstructed, latents = autoencoder(flat_activations)

            all_activations.append(flat_activations.cpu())
            all_latents.append(latents.cpu())

            # Reshape reconstructed back
            reconstructed = reconstructed.view(activations.shape)

            # 3) Transformer pass w/ reconstructed hidden states
            modified_outputs = transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
                custom_hidden_states={layer_idx: reconstructed}
            )
            reconstructed_loss = modified_outputs["loss"]

            original_losses.append(original_loss.item())
            reconstructed_losses.append(reconstructed_loss.item())

    avg_original_loss = float(np.mean(original_losses))
    avg_reconstructed_loss = float(np.mean(reconstructed_losses))

    activations_concat = torch.cat(all_activations, dim=0)
    latents_concat = torch.cat(all_latents, dim=0)

    interpretability_metrics = {
        "original_loss": avg_original_loss,
        "reconstructed_loss": avg_reconstructed_loss,
        "loss_difference": avg_reconstructed_loss - avg_original_loss,
        "neuron_sparsity": torch.mean((activations_concat > 0).float()).item(),
        "latent_sparsity": torch.mean((latents_concat > 0).float()).item(),
        "activation_entropy": compute_entropy(activations_concat),
        "latent_entropy": compute_entropy(latents_concat),
    }

    return interpretability_metrics

# -------------------------------
# 4) Main 'run' function
# -------------------------------
def run(out_dir: str):
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type="cuda")

    # GPT-2 config
    model_config = GPT2Config(
        n_layer=6,
        n_head=8,
        n_embd=512,     # Smaller dimension for demonstration
        vocab_size=50257
    )
    model = ModifiedGPT2(model_config).to(device)

    # ----------------------------------------
    # Load tiny subset from openwebtext
    # We'll select 10 samples for "train" and 10 for "val"
    # If you want more, increase the range size
    # ----------------------------------------
    train_dataset = OpenWebTextSubset(indices_range=range(10), max_length=128)
    val_dataset   = OpenWebTextSubset(indices_range=range(10, 20), max_length=128)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=2)

    # 1) Train Transformer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    with ctx:
        transformer_log = train_transformer(model, train_loader, optimizer, device, num_epochs=1000)

    # Save transformer checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, os.path.join(out_dir, "transformer_checkpoint.pt"))

    # 2) Train Autoencoder on intermediate layer
    layer_idx = model_config.n_layer // 2
    input_dim = model_config.n_embd
    latent_dim = input_dim // 2

    autoencoder = CompositionalSparseAutoencoder(
        input_dim=input_dim,
        feature_dim=256,
        composition_dim=128,
        n_heads=8,
        sparsity_k=32
    ).to(device)
    optimizer_ae = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)

    # Collect activations from training set
    activations = []
    model.eval()
    with torch.no_grad():
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device).float()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            # Middle-layer activations
            layer_activations = outputs["hidden_states"][layer_idx]  # [bsz, seq_len, embed_dim]
            activations.append(layer_activations.cpu())

    # Combine all
    activations = torch.cat(activations, dim=0).view(-1, input_dim)
    with ctx:
        autoencoder_log = train_autoencoder(autoencoder, activations, optimizer_ae, device=device, num_epochs=1000)

    torch.save({
        "model_state_dict": autoencoder.state_dict(),
        "optimizer_state_dict": optimizer_ae.state_dict(),
    }, os.path.join(out_dir, "autoencoder_checkpoint.pt"))

    # 3) Evaluate interpretability
    metrics = evaluate_interpretability(model, autoencoder, val_loader, layer_idx, device)

    # Prepare final info
    final_info = {
        "transformer_loss": transformer_log[-1]["loss"],
        "transformer_perplexity": float(torch.exp(torch.tensor(transformer_log[-1]["loss"]))),
        "autoencoder_loss": autoencoder_log[-1]["total_loss"],
        "reconstruction_error": metrics["loss_difference"],
        "downstream_loss_original": metrics["original_loss"],
        "downstream_loss_reconstructed": metrics["reconstructed_loss"],
        "downstream_loss_delta_absolute": metrics["loss_difference"],
        "downstream_loss_delta_percent": (
            metrics["loss_difference"] / metrics["original_loss"] * 100
            if metrics["original_loss"] != 0 else float("nan")
        ),
        "neuron_sparsity": metrics["neuron_sparsity"],
        "latent_sparsity": metrics["latent_sparsity"],
        "activation_entropy": metrics["activation_entropy"],
        "latent_entropy": metrics["latent_entropy"],
    }

    print("\n=== Final Info ===")
    for k, v in final_info.items():
        print(f"{k}: {v}")

    # Save logs
    results = {
        "transformer_training": transformer_log,
        "autoencoder_training": autoencoder_log,
        "interpretability_metrics": metrics
    }

    # with open(os.path.join(out_dir, "results.json"), "w") as f:
    #     json.dump(results, f, indent=2)

    with open(os.path.join(out_dir, "all_results.npy"), "wb") as f:
        np.save(f, results)

    # Optionally, you can also save final_info in a separate JSON
    with open(os.path.join(out_dir, "final_info.json"), "w") as f:
        json.dump(final_info, f, indent=2)

# -------------------------------
# 5) Entrypoint
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="openwebtext_test")
    args = parser.parse_args()
    run(args.out_dir)


