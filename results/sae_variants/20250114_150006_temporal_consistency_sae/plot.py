import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_feature_correlations(features, save_path):
    """Plot feature correlation matrix across positions."""
    # Compute correlation matrix
    corr_matrix = np.corrcoef(features.T)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Feature Activation Correlations Across Positions")
    plt.xlabel("Feature Index")
    plt.ylabel("Feature Index")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_temporal_consistency(activations, save_path):
    """Plot temporal consistency of feature activations."""
    window_size = activations.shape[0]
    time_points = np.arange(window_size)
    
    plt.figure(figsize=(10, 6))
    for i in range(min(20, activations.shape[1])):  # Plot first 20 features
        plt.plot(time_points, activations[:, i], alpha=0.5)
    
    plt.title("Temporal Consistency of Feature Activations")
    plt.xlabel("Time Step")
    plt.ylabel("Activation Value")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
