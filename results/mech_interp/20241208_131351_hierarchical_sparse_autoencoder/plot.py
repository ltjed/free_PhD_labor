import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np

def plot_attention_patterns(attention_weights: torch.Tensor, save_path: str):
    """Plot attention patterns between primary and secondary features."""
    attention_np = attention_weights.detach().cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_np, cmap='viridis')
    plt.title('Attention Patterns')
    plt.xlabel('Secondary Features')
    plt.ylabel('Primary Features')
    plt.savefig(save_path)
    plt.close()

def plot_feature_activations(primary_latents: torch.Tensor, secondary_latents: torch.Tensor, save_path: str):
    """Plot activation distributions for both levels."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    sns.histplot(primary_latents.detach().cpu().numpy().flatten(), ax=ax1)
    ax1.set_title('Primary Level Activations')
    
    sns.histplot(secondary_latents.detach().cpu().numpy().flatten(), ax=ax2)
    ax2.set_title('Secondary Level Activations')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
