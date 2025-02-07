import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_feature_correlations(W_dec, threshold, save_path):
    """Plot feature correlation matrix with threshold line"""
    # Compute correlations
    correlations = torch.mm(W_dec, W_dec.t()).cpu().numpy()
    np.fill_diagonal(correlations, 0)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    plt.imshow(correlations, cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation')
    
    # Add threshold lines
    plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=threshold, color='r', linestyle='--', alpha=0.5)
    
    plt.title(f'Feature Correlations (threshold={threshold:.2f})')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Index')
    
    # Save plot
    plt.savefig(save_path)
    plt.close()
