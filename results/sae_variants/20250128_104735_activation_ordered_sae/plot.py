import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def plot_activation_distribution(activation_frequencies, save_path):
    """Plot and save the activation frequency distribution."""
    plt.figure(figsize=(10, 6))
    
    # Sort frequencies in descending order
    sorted_freqs = np.sort(activation_frequencies.cpu().numpy())[::-1]
    
    # Create x-axis indices
    x = np.arange(len(sorted_freqs))
    
    # Plot distribution
    plt.plot(x, sorted_freqs)
    plt.xlabel('Feature Index (Sorted by Frequency)')
    plt.ylabel('Activation Frequency')
    plt.title('Feature Activation Distribution')
    plt.yscale('log')
    plt.grid(True)
    
    # Save plot
    plt.savefig(save_path)
    plt.close()

def plot_activation_history(activation_history, save_path):
    """Plot activation frequency changes over time."""
    plt.figure(figsize=(10, 6))
    
    # Convert history to numpy and transpose to get feature-wise trajectories
    history_array = torch.stack(activation_history).cpu().numpy()
    
    if len(activation_history) > 1:
        # Plot lines for top 10 and bottom 10 features
        mean_activations = history_array.mean(axis=0)
        top_indices = np.argsort(mean_activations)[-10:]
        bottom_indices = np.argsort(mean_activations)[:10]
        
        for idx in top_indices:
            plt.plot(history_array[:, idx], alpha=0.5, color='blue')
        for idx in bottom_indices:
            plt.plot(history_array[:, idx], alpha=0.5, color='red')
    else:
        plt.text(0.5, 0.5, 'Insufficient data for plotting', 
                ha='center', va='center', transform=plt.gca().transAxes)
        
    plt.xlabel('Training Step')
    plt.ylabel('Activation Frequency')
    plt.title('Feature Activation Trajectories\n(Top 10 blue, Bottom 10 red)')
    plt.yscale('log')
    plt.grid(True)
    
    plt.savefig(save_path)
    plt.close()
