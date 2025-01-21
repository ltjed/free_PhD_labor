import matplotlib.pyplot as plt
import numpy as np

def plot_feature_evolution(features, window_size, save_path=None):
    """Plot feature evolution over time windows."""
    plt.figure(figsize=(12, 6))
    plt.imshow(features.T, aspect='auto', cmap='viridis')
    plt.colorbar(label='Feature Activation')
    plt.xlabel('Time Window')
    plt.ylabel('Feature Index')
    plt.title(f'Feature Evolution (Window Size={window_size})')
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_temporal_stability(tss_scores, save_path=None):
    """Plot temporal stability scores."""
    plt.figure(figsize=(10, 5))
    plt.plot(tss_scores, '-o')
    plt.xlabel('Feature Index')
    plt.ylabel('Temporal Stability Score')
    plt.title('Feature Temporal Stability')
    if save_path:
        plt.savefig(save_path)
    plt.close()
