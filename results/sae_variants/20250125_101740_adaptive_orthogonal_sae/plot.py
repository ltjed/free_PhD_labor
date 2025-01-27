import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Style settings
plt.style.use('default')
plt.rcParams.update({
    'figure.figsize': [12, 8],
    'font.size': 12,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'axes.prop_cycle': plt.cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
})

# Define labels for each run
labels = {
    'run_0': 'Baseline (No Orthogonality)',
    'run_1': 'Fixed τ=0.1, Top-k=0.1%',
    'run_2': 'Fixed τ=0.2, Top-k=0.1%', 
    'run_3': 'Adaptive τ, Top-k=0.1%',
    'run_4': 'Adaptive τ, Top-k=0.5%'
}

def load_results(run_dir):
    """Load results from a run directory."""
    results_path = os.path.join(run_dir, 'all_results.npy')
    if os.path.exists(results_path):
        with open(results_path, 'rb') as f:
            return np.load(f, allow_pickle=True).item()
    return None

def plot_top_k_accuracies():
    """Plot top-k accuracies across runs."""
    k_values = [1, 2, 5, 10, 20, 50]
    accuracies = {run: [] for run in labels}
    
    for run in labels:
        results = load_results(run)
        if results and 'training_log' in results and results['training_log']:
            # Extract final top-k accuracies from the last non-empty log
            log = results['training_log'][-1]
            acc_values = []
            for k in k_values:
                # Use get() with default 0 to handle missing metrics
                acc_values.append(log.get(f'top_{k}_acc', 0))
            if any(acc_values):  # Only add if we have any non-zero values
                accuracies[run] = acc_values
    
    plt.figure(figsize=(12, 8))
    for run, acc_list in accuracies.items():
        if acc_list:  # Only plot if we have data
            plt.plot(k_values, acc_list, marker='o', label=labels[run])
    
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.title('Top-k Accuracy Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('top_k_accuracies.png')
    plt.close()

def plot_feature_correlations():
    """Plot feature correlation distributions."""
    plt.figure(figsize=(12, 8))
    
    for run in labels:
        results = load_results(run)
        if results and 'correlation_history' in results:
            correlations = results.get('correlation_history', [])
            if correlations:  # Only plot if we have data
                sns.kdeplot(data=correlations, label=labels[run])
    
    plt.xlabel('Feature Correlation')
    plt.ylabel('Density')
    plt.title('Feature Correlation Distribution')
    plt.legend()
    plt.grid(True)
    plt.savefig('feature_correlations.png')
    plt.close()

def plot_dataset_performance():
    """Plot performance across different datasets."""
    datasets = [
        'Helsinki-NLP/europarl',
        'Amazon Reviews',
        'Bias in Bios',
        'GitHub Code',
        'AG News'
    ]
    
    accuracies = {run: [] for run in labels}
    
    for run in labels:
        results = load_results(run)
        if results and 'final_info' in results:
            info = results['final_info']
            acc_values = []
            for dataset in datasets:
                # Use get() with default 0 to handle missing metrics
                acc_values.append(info.get(f'{dataset}_accuracy', 0))
            if any(acc_values):  # Only add if we have any non-zero values
                accuracies[run] = acc_values
    
    x = np.arange(len(datasets))
    width = 0.15
    
    plt.figure(figsize=(15, 8))
    for i, (run, acc_list) in enumerate(accuracies.items()):
        if acc_list:
            plt.bar(x + i*width, acc_list, width, label=labels[run])
    
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy')
    plt.title('Performance Across Datasets')
    plt.xticks(x + width*2, datasets, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('dataset_performance.png')
    plt.close()

def plot_tau_evolution():
    """Plot evolution of tau parameter for adaptive runs."""
    plt.figure(figsize=(12, 8))
    
    for run in ['run_3', 'run_4']:  # Only plot adaptive tau runs
        results = load_results(run)
        if results and 'tau_history' in results:
            tau_values = results.get('tau_history', [])
            if tau_values:  # Only plot if we have data
                steps = range(len(tau_values))
                plt.plot(steps, tau_values, label=labels[run])
    
    plt.xlabel('Training Step')
    plt.ylabel('τ Value')
    plt.title('Evolution of Adaptive τ')
    plt.legend()
    plt.grid(True)
    plt.savefig('tau_evolution.png')
    plt.close()

if __name__ == "__main__":
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Generate all plots
    plot_top_k_accuracies()
    plot_feature_correlations()
    plot_dataset_performance()
    plot_tau_evolution()
    
    print("Plots have been generated in the plots directory.")
