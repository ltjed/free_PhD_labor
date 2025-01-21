import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

# Configure plot style
plt.style.use('default')  # Use default style instead of seaborn
plt.rcParams.update({
    'figure.figsize': [12, 8],
    'figure.dpi': 100,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False
})
# Use a colorblind-friendly color palette
colors = ['#0077BB', '#EE7733', '#009988', '#CC3311', '#33BBEE']
sns.set_palette(colors)

# Dictionary mapping run directories to their display labels
labels = {
    'run_0': 'Baseline SAE',
    'run_1': 'TC-SAE Base',
    'run_2': 'TC-SAE (Adjusted Hyperparams)',
    'run_3': 'TC-SAE (Stability Improved)',
    'run_4': 'TC-SAE (Gradient Flow)'
}

def load_run_data(run_dir):
    """Load training data from a run directory."""
    results_path = os.path.join(run_dir, 'all_results.npy')
    if not os.path.exists(results_path):
        return None
    
    with open(results_path, 'rb') as f:
        data = np.load(f, allow_pickle=True).item()
    return data

def plot_training_metrics():
    """Plot training loss curves for all runs."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    for run_dir, label in labels.items():
        data = load_run_data(run_dir)
        if data is None:
            continue
            
        training_log = data.get('training_log', [])
        if not training_log:
            continue
            
        steps = range(len(training_log))
        
        # Plot reconstruction loss
        losses = [log.get('l2_loss', 0) for log in training_log]
        ax1.plot(steps, losses, label=label)
        
        # Plot sparsity loss
        sparsity = [log.get('sparsity_loss', 0) for log in training_log]
        ax2.plot(steps, sparsity, label=label)
        
        # Plot temporal loss (if available)
        temporal = [log.get('temporal_loss', 0) for log in training_log]
        ax3.plot(steps, temporal, label=label)
    
    ax1.set_title('Reconstruction Loss')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('L2 Loss')
    ax1.legend()
    
    ax2.set_title('Sparsity')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('L1 Loss')
    ax2.legend()
    
    ax3.set_title('Temporal Consistency')
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Temporal Loss')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

def plot_final_metrics():
    """Plot final evaluation metrics for all runs."""
    metrics = {
        'explained_variance': [],
        'mse': [],
        'kl_div': []
    }
    
    runs = []
    
    for run_dir, label in labels.items():
        data = load_run_data(run_dir)
        if data is None:
            continue
            
        final_info = data.get('final_info', {})
        if not final_info:
            continue
            
        runs.append(label)
        metrics['explained_variance'].append(final_info.get('explained_variance', 0))
        metrics['mse'].append(final_info.get('mse', 0))
        metrics['kl_div'].append(final_info.get('kl_div', 0))
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    x = np.arange(len(runs))
    width = 0.25
    
    ax1.bar(x, metrics['explained_variance'], width, label='Explained Variance')
    ax1.set_title('Explained Variance')
    ax1.set_xticks(x)
    ax1.set_xticklabels(runs, rotation=45)
    
    ax2.bar(x, metrics['mse'], width, label='MSE')
    ax2.set_title('Mean Squared Error')
    ax2.set_xticks(x)
    ax2.set_xticklabels(runs, rotation=45)
    
    ax3.bar(x, metrics['kl_div'], width, label='KL Divergence')
    ax3.set_title('KL Divergence')
    ax3.set_xticks(x)
    ax3.set_xticklabels(runs, rotation=45)
    
    plt.tight_layout()
    plt.savefig('final_metrics.png')
    plt.close()

def main():
    """Generate all plots."""
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Generate plots
    plot_training_metrics()
    plot_final_metrics()
    
    print("Plots have been generated in the plots directory:")
    print("1. training_metrics.png - Shows training progression")
    print("2. final_metrics.png - Shows final evaluation metrics")

if __name__ == "__main__":
    main()
