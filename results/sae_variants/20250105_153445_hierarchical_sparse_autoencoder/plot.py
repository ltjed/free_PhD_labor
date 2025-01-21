import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns

def plot_hierarchical_features(coarse_features, fine_features, coarse_to_fine_weights, save_path=None):
    """Plot hierarchical feature relationships."""
    plt.figure(figsize=(12, 8))
    
    # Create heatmap of coarse-to-fine weights
    sns.heatmap(coarse_to_fine_weights.cpu().numpy(), 
                cmap='RdBu_r',
                center=0,
                xticklabels=False,
                yticklabels=False)
    
    plt.title('Hierarchical Feature Relationships')
    plt.xlabel('Fine Features')
    plt.ylabel('Coarse Features')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_activation_patterns(coarse_acts, fine_acts, save_path=None):
    """Plot activation patterns at both levels."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot coarse activations
    sns.heatmap(coarse_acts.cpu().numpy()[:100].T, 
                ax=ax1, 
                cmap='viridis',
                xticklabels=False,
                yticklabels=False)
    ax1.set_title('Coarse Feature Activations')
    
    # Plot fine activations
    sns.heatmap(fine_acts.cpu().numpy()[:100].T,
                ax=ax2,
                cmap='viridis',
                xticklabels=False,
                yticklabels=False)
    ax2.set_title('Fine Feature Activations')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()
