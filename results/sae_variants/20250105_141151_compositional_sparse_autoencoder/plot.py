import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np

def plot_attention_patterns(attention_weights, save_path=None):
    """Plot attention patterns from the composition network."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights.detach().cpu().numpy(), cmap='viridis')
    plt.title('Feature Composition Attention Patterns')
    plt.xlabel('Key Features')
    plt.ylabel('Query Features')
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_feature_reuse(activation_counts, save_path=None):
    """Plot feature reuse statistics."""
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(activation_counts)), sorted(activation_counts, reverse=True))
    plt.title('Feature Reuse Distribution')
    plt.xlabel('Feature Index (sorted)')
    plt.ylabel('Activation Count')
    if save_path:
        plt.savefig(save_path)
    plt.close()

def visualize_compositions(sae_model, activations, save_dir):
    """Generate visualizations for feature compositions."""
    with torch.no_grad():
        # Get attention patterns
        feat = torch.relu((activations - sae_model.b_dec) @ sae_model.W_feat + sae_model.b_feat)
        _, attn_weights = sae_model.mha(feat.unsqueeze(1), feat.unsqueeze(1), feat.unsqueeze(1))
        
        # Plot attention patterns
        plot_attention_patterns(attn_weights[0], f"{save_dir}/attention_patterns.png")
        
        # Get feature activation statistics
        encoded = sae_model.encode(activations)
        activation_counts = (encoded > 0).float().sum(dim=0)
        plot_feature_reuse(activation_counts, f"{save_dir}/feature_reuse.png")
