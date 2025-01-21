import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Dictionary mapping run directories to their display labels
labels = {
    "run_0": "Baseline SAE",
    "run_1": "Initial Temporal SAE",
    "run_2": "Fixed Training Loop",
    "run_3": "LR Schedule",
    "run_4": "Layer Norm",
    "run_5": "GELU Activation"
}

def load_results(run_dir):
    """Load results from a run directory."""
    info_path = os.path.join(run_dir, "final_info.json")
    if not os.path.exists(info_path):
        return None
    
    with open(info_path, 'r') as f:
        return json.load(f)

def plot_metrics():
    """Generate plots comparing key metrics across runs."""
    # Metrics to plot
    metrics = {
        'Reconstruction Quality': ['explained_variance', 'cosine_similarity'],
        'Sparsity': ['l0', 'l1'],
        'Model Preservation': ['ce_loss_score']
    }
    
    # Setup the plot grid
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 12))
    fig.suptitle('SAE Performance Metrics Across Runs', fontsize=14)
    
    # Colors for consistent run representation
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    
    for idx, (metric_group, metric_list) in enumerate(metrics.items()):
        ax = axes[idx]
        x = np.arange(len(labels))
        width = 0.35
        
        for i, metric in enumerate(metric_list):
            values = []
            for run_dir in labels.keys():
                results = load_results(run_dir)
                if results and metric in results:
                    values.append(results[metric])
                else:
                    values.append(0)  # Default value if metric not found
            
            offset = width * (i - len(metric_list)/2 + 0.5)
            bars = ax.bar(x + offset, values, width, label=metric)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom')
        
        ax.set_title(metric_group)
        ax.set_xticks(x)
        ax.set_xticklabels(labels.values(), rotation=45)
        ax.legend()
        
        # Add grid for better readability
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_curves():
    """Plot training curves from the logs."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    for run_dir, label in labels.items():
        log_path = os.path.join(run_dir, "all_results.npy")
        if not os.path.exists(log_path):
            continue
            
        with open(log_path, 'rb') as f:
            results = np.load(f, allow_pickle=True).item()
            
        if 'training_log' in results:
            log = results['training_log']
            steps = range(len(log))
            
            # Plot reconstruction loss
            losses = [entry.get('l2_loss', 0) for entry in log]
            ax1.plot(steps, losses, label=label)
            
            # Plot sparsity
            sparsity = [entry.get('sparsity_loss', 0) for entry in log]
            ax2.plot(steps, sparsity, label=label)
    
    ax1.set_title('Reconstruction Loss Over Training')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('L2 Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_title('Sparsity Loss Over Training')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('L1 Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Create plots
    plot_metrics()
    plot_training_curves()
