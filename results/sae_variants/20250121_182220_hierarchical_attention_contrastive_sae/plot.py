import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_attention_weights_over_time(weight_history, save_dir):
    """Plot how attention weights for each layer evolved during training."""
    weights = torch.stack(weight_history)
    plt.figure(figsize=(10, 6))
    labels = ['L-1', 'L', 'L+1']
    
    for i in range(weights.shape[1]):
        plt.plot(weights[:, i].numpy(), label=labels[i])
    
    plt.xlabel('Training Steps')
    plt.ylabel('Attention Weight')
    plt.title('Evolution of Layer Attention Weights')
    plt.legend()
    plt.grid(True)
    plt.savefig(Path(save_dir) / 'attention_weights_evolution.png')
    plt.close()

def plot_feature_activation_patterns(sae_model, activations, save_dir):
    """Plot heatmap of feature activations across samples."""
    with torch.no_grad():
        _, features = sae_model(activations, output_features=True)
    
    # Take a subset of features for visualization
    features = features[:100, :100].cpu().numpy()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(features, cmap='viridis')
    plt.title('Feature Activation Patterns')
    plt.xlabel('Feature Index')
    plt.ylabel('Sample Index')
    plt.savefig(Path(save_dir) / 'feature_activation_patterns.png')
    plt.close()

def plot_layer_contributions(sae_model, activations, save_dir):
    """Plot the relative contributions of each attention layer."""
    with torch.no_grad():
        # Get attention weights for each layer
        prev_attn = torch.randn_like(activations)  # Simulate attention
        curr_attn = torch.randn_like(activations)
        next_attn = torch.randn_like(activations)
        
        # Forward pass to get layer contributions
        sae_model.encode(activations, prev_attn, curr_attn, next_attn)
        weights = torch.softmax(sae_model.attn_combine, dim=0).cpu().numpy()
    
    plt.figure(figsize=(8, 6))
    labels = ['L-1', 'L', 'L+1']
    plt.bar(labels, weights)
    plt.title('Relative Layer Contributions')
    plt.ylabel('Contribution Weight')
    plt.savefig(Path(save_dir) / 'layer_contributions.png')
    plt.close()

def generate_analysis_plots(sae_model, activations, weight_history, save_dir):
    """Generate all analysis plots."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    plot_attention_weights_over_time(weight_history, save_dir)
    plot_feature_activation_patterns(sae_model, activations, save_dir)
    plot_layer_contributions(sae_model, activations, save_dir)
