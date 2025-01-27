import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

# Dictionary mapping run numbers to their descriptive labels
labels = {
    "run_5": "Gradient-Based Selection",
    "run_6": "Adaptive Pair Count",
    "run_7": "Aggressive Orthogonality",
    "run_8": "Temporal Correlation",
    "run_9": "Enhanced Temporal",
    "run_10": "Mutual Information"
}

def load_run_data(run_dir):
    """Load training data from a run directory."""
    results_path = os.path.join(run_dir, "all_results.npy")
    if not os.path.exists(results_path):
        return None
    
    with open(results_path, 'rb') as f:
        data = np.load(f, allow_pickle=True).item()
    return data

def plot_training_metrics(runs_data):
    """Plot training metrics across runs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for run_name, data in runs_data.items():
        if data is None:
            continue
            
        training_log = data.get('training_log', [])
        if not training_log:
            continue
            
        steps = range(len(training_log))
        l2_losses = [log.get('l2_loss', np.nan) for log in training_log]
        sparsity = [log.get('sparsity_loss', np.nan) for log in training_log]
        
        label = labels.get(run_name, run_name)
        ax1.plot(steps, l2_losses, label=label, alpha=0.7)
        ax2.plot(steps, sparsity, label=label, alpha=0.7)
    
    ax1.set_title('Reconstruction Loss Over Training')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('L2 Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_title('Sparsity Over Training')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('L1 Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

def plot_feature_correlations(runs_data):
    """Plot feature correlation heatmaps for final states."""
    n_runs = len(runs_data)
    if n_runs == 0:
        return
        
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (run_name, data) in enumerate(runs_data.items()):
        if data is None or idx >= len(axes):
            continue
            
        # Extract final feature activations
        if 'training_log' in data and len(data['training_log']) > 0:
            final_log = data['training_log'][-1]
            if hasattr(final_log, 'f'):
                features = final_log.f.detach().cpu().numpy()
                corr = np.corrcoef(features.T)
                
                sns.heatmap(corr, ax=axes[idx], cmap='coolwarm', center=0,
                           vmin=-1, vmax=1, square=True)
                axes[idx].set_title(f'{labels.get(run_name, run_name)}\nFeature Correlations')
                axes[idx].set_xticks([])
                axes[idx].set_yticks([])
    
    plt.tight_layout()
    plt.savefig('feature_correlations.png')
    plt.close()

def main():
    # Load data from all runs
    runs_data = {}
    for run_name in labels.keys():
        data = load_run_data(run_name)
        if data is not None:
            runs_data[run_name] = data
    
    # Generate plots
    plot_training_metrics(runs_data)
    plot_feature_correlations(runs_data)

if __name__ == "__main__":
    main()
