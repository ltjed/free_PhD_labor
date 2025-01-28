import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import json
from collections import defaultdict

# Dictionary mapping run numbers to descriptive labels
labels = {
    "0": "Baseline",
    "8": "LR Schedule", 
    "9": "Adaptive Penalty"
}

def load_results(run_dir):
    """Load results from a run directory."""
    path = os.path.join(run_dir, "final_info.json")
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

def plot_metrics_comparison():
    """Plot comparison of key metrics across runs."""
    metrics = defaultdict(list)
    runs = []
    
    # Collect metrics from each run
    for run_num in sorted(labels.keys()):
        run_dir = f"run_{run_num}"
        results = load_results(run_dir)
        if results:
            runs.append(run_num)
            for layer_results in results.values():
                if 'final_info' in layer_results:
                    info = layer_results['final_info']
                    metrics['loss'].append(info['final_loss'])
                    
                if 'core evaluation results' in layer_results:
                    core = layer_results['core evaluation results']['metrics']
                    metrics['kl_div'].append(core['model_behavior_preservation']['kl_div_score'])
                    metrics['mse'].append(core['reconstruction_quality']['mse'])
                    metrics['l0_sparsity'].append(core['sparsity']['l0'])
                    metrics['l1_sparsity'].append(core['sparsity']['l1'])
                    metrics['cosine_sim'].append(core['reconstruction_quality']['cossim'])

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Comparison of Key Metrics Across Runs', fontsize=16)

    # Helper function to safely plot metrics
    def safe_plot_bars(ax, x, metrics, key1, key2, label1, label2, title):
        if len(metrics.get(key1, [])) > 0:
            ax.bar(x - 0.2, metrics[key1], width=0.4, label=label1)
        if len(metrics.get(key2, [])) > 0:
            ax.bar(x + 0.2, metrics[key2], width=0.4, label=label2)
        ax.set_xticks(x)
        ax.set_xticklabels([labels[run] for run in runs])
        if len(metrics.get(key1, [])) > 0 or len(metrics.get(key2, [])) > 0:
            ax.legend()
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    # Plot metrics if available
    x = np.arange(len(runs))
    
    # Plot loss and KL divergence
    safe_plot_bars(axes[0,0], x, metrics, 
                  'loss', 'kl_div',
                  'Final Loss', 'KL Divergence',
                  'Loss and Model Preservation')
    
    # Plot sparsity metrics
    safe_plot_bars(axes[0,1], x, metrics,
                  'l0_sparsity', 'l1_sparsity',
                  'L0 Sparsity', 'L1 Sparsity', 
                  'Sparsity Metrics')
    
    # Plot reconstruction metrics
    safe_plot_bars(axes[1,0], x, metrics,
                  'mse', 'cosine_sim',
                  'MSE', 'Cosine Similarity',
                  'Reconstruction Quality')

    # Add summary text
    ax = axes[1,1]
    ax.axis('off')
    summary = (
        "Summary of Improvements:\n\n"
        "1. Learning Rate Scheduling (Run 8):\n"
        "   - Improved training stability\n"
        "   - Better feature sparsification\n\n"
        "2. Adaptive Penalty (Run 9):\n"
        "   - Dynamic L1 penalty scaling\n"
        "   - Enhanced reconstruction quality\n"
        "   - Maintained model behavior"
    )
    ax.text(0.1, 0.1, summary, fontsize=10, va='top')

    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/metrics_comparison.png')
    plt.close()

if __name__ == "__main__":
    plot_metrics_comparison()

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
