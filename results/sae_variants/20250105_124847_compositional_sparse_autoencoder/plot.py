import matplotlib.pyplot as plt
import torch
import seaborn as sns
import numpy as np

def plot_attention_patterns(attention_weights, save_path=None):
    """Plot attention patterns between features."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights.detach().cpu().numpy(), cmap='viridis')
    plt.title('Feature Composition Attention Patterns')
    plt.xlabel('Key Features')
    plt.ylabel('Query Features')
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_feature_reuse(feature_counts, save_path=None):
    """Plot histogram of feature activation frequencies."""
    plt.figure(figsize=(10, 6))
    plt.hist(feature_counts, bins=50)
    plt.title('Feature Reuse Distribution')
    plt.xlabel('Activation Count')
    plt.ylabel('Number of Features')
    if save_path:
        plt.savefig(save_path)
    plt.close()
