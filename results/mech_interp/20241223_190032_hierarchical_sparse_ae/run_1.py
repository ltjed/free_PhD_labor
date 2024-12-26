import os
import json
import torch
import wandb
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import CrossEntropyLoss
from contextlib import nullcontext
from typing import Dict, List
from datasets import load_dataset

class HierarchicalSparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        # Bottom layer (90% sparsity target)
        self.encoder1 = nn.Linear(input_dim, hidden_dim)
        self.decoder1 = nn.Linear(hidden_dim, input_dim)
        self.gate1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Top layer (70% sparsity target)
        self.encoder2 = nn.Linear(hidden_dim, latent_dim)
        self.decoder2 = nn.Linear(latent_dim, hidden_dim)
        self.gate2 = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.Sigmoid()
        )
        
        self.training_phase = 1  # 1: bottom, 2: top, 3: fine-tune
        
    def get_sparsity_targets(self):
        return {
            "bottom": 0.90,  # 90% sparsity
            "top": 0.70      # 70% sparsity
        }
        
    def forward(self, x):
        # Bottom layer
        h1 = F.relu(self.encoder1(x))
        g1 = self.gate1(x)
        h1_gated = h1 * g1
        
        # Top layer
        h2 = F.relu(self.encoder2(h1_gated))
        g2 = self.gate2(h1_gated)
        h2_gated = h2 * g2
        
        # Decoding
        if self.training_phase == 1:
            # Only use bottom layer
            recon = self.decoder1(h1_gated)
            return recon, h1_gated, None
        else:
            # Use both layers
            h1_recon = self.decoder2(h2_gated)
            skip_connection = h1_gated if self.training_phase == 2 else h1_gated.detach()
            recon = self.decoder1(h1_recon + skip_connection)
            return recon, h1_gated, h2_gated

class ModifiedGPT2(GPT2LMHeadModel):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        custom_hidden_states=None,
        output_hidden_states=False,
    ):
        if custom_hidden_states is None:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=output_hidden_states,
            )
        
        # Get the initial embeddings
        inputs_embeds = self.transformer.wte(input_ids)
        position_embeds = self.transformer.wpe(torch.arange(input_ids.size(1), device=input_ids.device))
        hidden_states = inputs_embeds + position_embeds
        
        # Initialize lists to store all hidden states if needed
        all_hidden_states = () if output_hidden_states else None
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # Run through the layers
        for i, block in enumerate(self.transformer.h):
            # If we have custom hidden states for this layer, use them
            if i in custom_hidden_states:
                hidden_states = custom_hidden_states[i]
            else:
                # Standard layer forward pass
                layer_outputs = block(
                    hidden_states,
                    attention_mask=attention_mask,
                )
                hidden_states = layer_outputs[0]
            
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
        
        # Final layer norm
        hidden_states = self.transformer.ln_f(hidden_states)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # Get logits
        lm_logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            hidden_states=all_hidden_states if output_hidden_states else None,
        )

class OpenWebTextDataset(torch.utils.data.Dataset):
    def __init__(self, split, max_length=128):
        self.dataset = load_dataset("openwebtext", split=split)
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]["text"]
        return {"input_ids": torch.tensor(text[:self.max_length])}

def compute_entropy(tensor: torch.Tensor) -> float:
    # Flatten all dimensions except the feature dimension
    flat_tensor = tensor.view(-1, tensor.size(-1))
    # Compute empirical distribution
    hist = torch.histc(flat_tensor, bins=100)
    probs = hist / hist.sum()
    # Remove zero probabilities before computing log
    probs = probs[probs > 0]
    return -(probs * torch.log(probs)).sum().item()

def train_transformer(model, train_loader, optimizer, device, num_epochs):
    model.train()
    log = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        log.append({"epoch": epoch, "loss": avg_loss})
        wandb.log({"transformer_loss": avg_loss})
    
    return log

def train_autoencoder(autoencoder, activations, optimizer, num_epochs, device, l1_coefficient=0.1):
    autoencoder.train()
    log = []
    
    # Training phases
    phase_epochs = {
        1: num_epochs // 3,  # Bottom layer
        2: num_epochs // 3,  # Top layer
        3: num_epochs // 3   # Fine-tuning
    }
    
    current_epoch = 0
    for phase in [1, 2, 3]:
        autoencoder.training_phase = phase
        
        # Freeze/unfreeze parameters based on phase
        if phase == 1:
            for param in autoencoder.encoder2.parameters(): param.requires_grad = False
            for param in autoencoder.decoder2.parameters(): param.requires_grad = False
            for param in autoencoder.gate2.parameters(): param.requires_grad = False
        elif phase == 2:
            for param in autoencoder.encoder1.parameters(): param.requires_grad = False
            for param in autoencoder.decoder1.parameters(): param.requires_grad = False
            for param in autoencoder.gate1.parameters(): param.requires_grad = False
            for param in autoencoder.encoder2.parameters(): param.requires_grad = True
            for param in autoencoder.decoder2.parameters(): param.requires_grad = True
            for param in autoencoder.gate2.parameters(): param.requires_grad = True
        else:  # phase 3
            for param in autoencoder.parameters(): param.requires_grad = True
            
        for epoch in range(phase_epochs[phase]):
            reconstruction, h1, h2 = autoencoder(activations)
            
            # Reconstruction loss
            recon_loss = F.mse_loss(reconstruction, activations)
            
            # Sparsity targets
            sparsity_targets = autoencoder.get_sparsity_targets()
            
            # L1 losses for sparsity
            l1_loss_bottom = l1_coefficient * torch.mean(torch.abs(h1))
            l1_loss_top = l1_coefficient * torch.mean(torch.abs(h2)) if h2 is not None else 0
            
            # Adjust L1 penalties based on current sparsity levels
            bottom_sparsity = (h1 == 0).float().mean()
            top_sparsity = (h2 == 0).float().mean() if h2 is not None else torch.tensor(0.)
            
            l1_bottom_scale = (1 - (bottom_sparsity / sparsity_targets["bottom"])).clamp(min=0)
            l1_top_scale = (1 - (top_sparsity / sparsity_targets["top"])).clamp(min=0) if h2 is not None else 0
            
            # Total loss
            total_loss = recon_loss + l1_bottom_scale * l1_loss_bottom + l1_top_scale * l1_loss_top
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            metrics = {
                "epoch": current_epoch,
                "phase": phase,
                "total_loss": total_loss.item(),
                "recon_loss": recon_loss.item(),
                "l1_loss_bottom": l1_loss_bottom.item(),
                "bottom_sparsity": bottom_sparsity.item(),
            }
            
            if h2 is not None:
                metrics.update({
                    "l1_loss_top": l1_loss_top.item(),
                    "top_sparsity": top_sparsity.item(),
                })
            
            log.append(metrics)
            wandb.log(metrics)
            
            current_epoch += 1
    
    return log

def evaluate_interpretability(
    transformer: nn.Module,
    autoencoder: HierarchicalSparseAutoencoder,
    val_loader: DataLoader,
    layer_idx: int,
    device: torch.device
) -> Dict:
    transformer.eval()
    autoencoder.eval()
    
    original_losses = []
    reconstructed_losses = []
    all_activations = []
    all_latents = []
    reconstruction_errors = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Get original loss and activations
            outputs = transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
                output_hidden_states=True
            )
            original_loss = outputs.loss
            original_losses.append(original_loss.item())
            activations = outputs.hidden_states[layer_idx]
            
            # Get reconstructed activations
            flat_activations = activations.view(-1, activations.size(-1))
            reconstructed, latents = autoencoder(flat_activations)
            reconstructed = reconstructed.view(activations.shape)
            
            # Calculate reconstruction error
            reconstruction_error = F.mse_loss(reconstructed, activations)
            reconstruction_errors.append(reconstruction_error.item())
            
            # Replace original activations with reconstructed ones and get new loss
            modified_outputs = transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
                custom_hidden_states={layer_idx: reconstructed}
            )
            reconstructed_loss = modified_outputs.loss
            reconstructed_losses.append(reconstructed_loss.item())
            
            all_activations.append(activations.cpu())
            all_latents.append(latents.cpu())
    
    # Compute average losses and metrics
    avg_original_loss = sum(original_losses) / len(original_losses)
    avg_reconstructed_loss = sum(reconstructed_losses) / len(reconstructed_losses)
    avg_reconstruction_error = sum(reconstruction_errors) / len(reconstruction_errors)
    
    # Calculate loss deltas
    downstream_loss_delta = avg_reconstructed_loss - avg_original_loss
    downstream_loss_percent = (downstream_loss_delta / avg_original_loss) * 100
    
    # Aggregate activation and latent statistics
    activations = torch.cat(all_activations, dim=0)
    latents = torch.cat(all_latents, dim=0)
    
    metrics = {
        "downstream_loss_original": avg_original_loss,
        "downstream_loss_reconstructed": avg_reconstructed_loss,
        "downstream_loss_delta_absolute": downstream_loss_delta,
        "downstream_loss_delta_percent": downstream_loss_percent,
        "reconstruction_error": avg_reconstruction_error,
        "neuron_sparsity": torch.mean((activations > 0).float()).item(),
        "activation_entropy": compute_entropy(activations),
        "latent_sparsity": torch.mean((latents > 0).float()).item(),
        "latent_entropy": compute_entropy(latents)
    }
    
    return metrics

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

    model = ModifiedGPT2(model_config).to(device)
    
    train_dataset = OpenWebTextDataset("train", max_length=128)
    val_dataset = OpenWebTextDataset("validation", max_length=128)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Train transformer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    with ctx:
        transformer_log = train_transformer(model, train_loader, optimizer, device, num_epochs=10)

    # Save transformer checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(out_dir, 'transformer_checkpoint.pt'))
    
    # Train hierarchical autoencoder on intermediate layer activations
    layer_idx = model_config.n_layer // 2  # Use middle layer
    input_dim = model_config.n_embd
    hidden_dim = input_dim // 2
    latent_dim = hidden_dim // 2
    
    autoencoder = HierarchicalSparseAutoencoder(input_dim, hidden_dim, latent_dim).to(device)
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
    with ctx:
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
        "transformer_perplexity": torch.exp(torch.tensor(transformer_log[-1]["loss"])).item(),
        "autoencoder_loss": autoencoder_log[-1]["total_loss"],
        "reconstruction_error": metrics["reconstruction_error"],
        "downstream_loss_original": metrics["downstream_loss_original"],
        "downstream_loss_reconstructed": metrics["downstream_loss_reconstructed"],
        "downstream_loss_delta_absolute": metrics["downstream_loss_delta_absolute"],
        "downstream_loss_delta_percent": metrics["downstream_loss_delta_percent"],
        "neuron_sparsity": metrics["neuron_sparsity"],
        "latent_sparsity": metrics["latent_sparsity"],
        "activation_entropy": metrics["activation_entropy"],
        "latent_entropy": metrics["latent_entropy"]
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="run_0")
    args = parser.parse_args()
    run(args.out_dir)
