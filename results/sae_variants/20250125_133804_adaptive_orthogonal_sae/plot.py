import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Dictionary mapping run numbers to their descriptive labels
labels = {
    "run_0": "Baseline",
    "run_1": "Fixed τ",
    "run_2": "Adaptive τ", 
    "run_3": "Aggressive τ",
    "run_4": "Global SVD",
    "run_5": "Gradient Projection"
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
    """Plot training metrics across all runs."""
    plt.figure(figsize=(15, 10))
    
    # Plot loss curves
    plt.subplot(2, 2, 1)
    for run_name, data in runs_data.items():
        if data and 'training_log' in data:
            losses = [log.get('loss', None) for log in data['training_log'] if isinstance(log, dict)]
            plt.plot(losses, label=labels[run_name])
    plt.title('Training Loss Over Time')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot orthogonality metrics
    plt.subplot(2, 2, 2)
    for run_name, data in runs_data.items():
        if data and 'training_log' in data:
            ortho_losses = [log.get('ortho_loss', None) for log in data['training_log'] if isinstance(log, dict)]
            if any(x is not None for x in ortho_losses):
                plt.plot(ortho_losses, label=labels[run_name])
    plt.title('Orthogonality Loss Over Time')
    plt.xlabel('Training Steps')
    plt.ylabel('Orthogonality Loss')
    plt.legend()
    plt.grid(True)

    # Plot tau values
    plt.subplot(2, 2, 3)
    for run_name, data in runs_data.items():
        if data and 'training_log' in data:
            taus = [log.get('tau', None) for log in data['training_log'] if isinstance(log, dict)]
            if any(x is not None for x in taus):
                plt.plot(taus, label=labels[run_name])
    plt.title('τ Values Over Time')
    plt.xlabel('Training Steps')
    plt.ylabel('τ')
    plt.legend()
    plt.grid(True)

    # Plot sparsity metrics
    plt.subplot(2, 2, 4)
    for run_name, data in runs_data.items():
        if data and 'training_log' in data:
            sparsity = [log.get('sparsity_loss', None) for log in data['training_log'] if isinstance(log, dict)]
            plt.plot(sparsity, label=labels[run_name])
    plt.title('Sparsity Loss Over Time')
    plt.xlabel('Training Steps')
    plt.ylabel('Sparsity Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_final_metrics(runs_data):
    """Plot final metrics comparison across runs."""
    metrics = ['loss', 'ortho_loss', 'sparsity_loss']
    final_values = {metric: [] for metric in metrics}
    run_names = []
    
    for run_name, data in runs_data.items():
        if data and 'training_log' in data and data['training_log']:
            run_names.append(labels[run_name])
            final_log = data['training_log'][-1]
            for metric in metrics:
                final_values[metric].append(final_log.get(metric, 0))
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(run_names))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        plt.bar(x + i*width, final_values[metric], width, label=metric.replace('_', ' ').title())
    
    plt.xlabel('Runs')
    plt.ylabel('Value')
    plt.title('Final Metrics Comparison')
    plt.xticks(x + width, run_names, rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('final_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Set style
    plt.style.use('bmh')  # Using a built-in matplotlib style
    
    # Load data from all runs
    runs_data = {}
    for run_name in labels.keys():
        data = load_run_data(run_name)
        if data:
            runs_data[run_name] = data
    
    # Generate plots
    plot_training_metrics(runs_data)
    plot_final_metrics(runs_data)

if __name__ == "__main__":
    main()
