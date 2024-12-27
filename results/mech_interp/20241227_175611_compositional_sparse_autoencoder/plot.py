import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_attention_patterns(attention_weights, save_path):
    """Plot attention patterns from the composition network."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attention_weights.detach().cpu().numpy(), cmap='viridis')
    plt.colorbar(im)
    plt.title('Feature Composition Attention Patterns')
    plt.xlabel('Features')
    plt.ylabel('Compositions')
    plt.savefig(save_path)
    plt.close()

def plot_feature_reuse(feature_usage, save_path):
    """Plot feature reuse statistics."""
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(feature_usage)), sorted(feature_usage, reverse=True))
    plt.title('Feature Reuse Distribution')
    plt.xlabel('Feature Index')
    plt.ylabel('Usage Count')
    plt.savefig(save_path)
    plt.close()

def plot_sparsity_evolution(sparsity_log, save_path):
    """Plot how sparsity changes during training."""
    plt.figure(figsize=(10, 6))
    plt.plot(sparsity_log)
    plt.title('Sparsity Evolution During Training')
    plt.xlabel('Training Steps')
    plt.ylabel('Active Features Ratio')
    plt.savefig(save_path)
    plt.close()
