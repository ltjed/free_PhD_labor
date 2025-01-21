import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List, Dict

def plot_feature_evolution(features: torch.Tensor, window_size: int, save_path: str):
    """Plot feature evolution over the sliding window."""
    plt.figure(figsize=(12, 8))
    plt.imshow(features.cpu().numpy(), aspect='auto', cmap='viridis')
    plt.colorbar(label='Feature Activation')
    plt.xlabel('Time Step')
    plt.ylabel('Feature Index')
    plt.title(f'Feature Evolution (Window Size={window_size})')
    plt.savefig(save_path)
    plt.close()

def plot_feature_lifecycle(lifecycle_stats: Dict, save_path: str):
    """Plot feature lifecycle statistics."""
    births = lifecycle_stats['births']
    deaths = lifecycle_stats['deaths']
    lifetimes = lifecycle_stats['lifetimes']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Birth/Death events
    ax1.plot(births, label='Births')
    ax1.plot(deaths, label='Deaths')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Count')
    ax1.set_title('Feature Birth/Death Events')
    ax1.legend()
    
    # Lifetime distribution
    ax2.hist(lifetimes, bins=30)
    ax2.set_xlabel('Lifetime Duration')
    ax2.set_ylabel('Count')
    ax2.set_title('Feature Lifetime Distribution')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
