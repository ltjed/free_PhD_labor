import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Style settings
plt.style.use('default')  # Use matplotlib default style
plt.rcParams.update({
    'figure.figsize': [12, 8],
    'font.size': 12,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False
})
sns.set_palette("husl")

# Dictionary mapping run names to their display labels
labels = {
    'run_0': 'Baseline SAE',
    'run_1': 'Initial Temporal-Relational',
    'run_2': 'Basic SAE Implementation',
    'run_3': 'Layer Norm + LR Adjust',
    'run_4': 'Gradient Clip + Skip Conn',
    'run_5': 'Hierarchical SAE'
}

def load_results(run_dir):
    """Load results from a run directory."""
    info_path = os.path.join(run_dir, 'final_info.json')
    if os.path.exists(info_path):
        try:
            with open(info_path, 'r') as f:
                data = json.load(f)
                # Ensure metrics exist
                if 'metrics' not in data:
                    print(f"Warning: No metrics found in {run_dir}")
                    data['metrics'] = {
                        'reconstruction_quality': {
                            'explained_variance': 0,
                            'mse': 0,
                            'cossim': 0
                        }
                    }
                return data
        except json.JSONDecodeError:
            print(f"Error: Could not parse JSON from {info_path}")
            return None
    print(f"Warning: No results file found at {info_path}")
    return None

def plot_reconstruction_metrics(results_by_run):
    """Plot reconstruction quality metrics."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    runs = list(results_by_run.keys())
    x_positions = np.arange(len(runs))
    
    # Extract metrics safely with error handling
    explained_var = []
    mse = []
    cossim = []
    
    for results in results_by_run.values():
        if isinstance(results, dict):
            metrics = results.get('metrics', {})
            if isinstance(metrics, dict):
                recon_quality = metrics.get('reconstruction_quality', {})
                explained_var.append(recon_quality.get('explained_variance', 0))
                mse.append(recon_quality.get('mse', 0))
                cossim.append(recon_quality.get('cossim', 0))
            else:
                explained_var.append(0)
                mse.append(0)
                cossim.append(0)
    
    # Explained Variance
    ax1.bar(x_positions, explained_var)
    ax1.set_title('Explained Variance')
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels([labels.get(run, run) for run in runs], rotation=45)
    
    # MSE
    ax2.bar(x_positions, mse)
    ax2.set_title('Mean Squared Error')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels([labels.get(run, run) for run in runs], rotation=45)
    
    # Cosine Similarity
    ax3.bar(x_positions, cossim)
    ax3.set_title('Cosine Similarity')
    ax3.set_xticks(x_positions)
    ax3.set_xticklabels([labels.get(run, run) for run in runs], rotation=45)
    
    plt.tight_layout()
    plt.savefig('reconstruction_metrics.png')
    plt.close()

def plot_sparsity_metrics(results_by_run):
    """Plot sparsity metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    runs = list(results_by_run.keys())
    x_positions = np.arange(len(runs))
    
    # Extract metrics safely with error handling
    l0_sparsity = []
    l1_sparsity = []
    
    for results in results_by_run.values():
        if isinstance(results, dict):
            metrics = results.get('metrics', {})
            sparsity = metrics.get('sparsity', {})
            l0_sparsity.append(sparsity.get('l0', 0))
            l1_sparsity.append(sparsity.get('l1', 0))
        else:
            l0_sparsity.append(0)
            l1_sparsity.append(0)
    
    # L0 Sparsity
    ax1.bar(x_positions, l0_sparsity)
    ax1.set_title('L0 Sparsity')
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels([labels.get(run, run) for run in runs], rotation=45)
    
    # L1 Sparsity
    ax2.bar(x_positions, l1_sparsity)
    ax2.set_title('L1 Sparsity')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels([labels.get(run, run) for run in runs], rotation=45)
    
    plt.tight_layout()
    plt.savefig('sparsity_metrics.png')
    plt.close()

def plot_model_preservation(results_by_run):
    """Plot model behavior preservation metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    runs = list(results_by_run.keys())
    
    # Safely extract metrics with error handling
    kl_div = []
    ce_loss = []
    for results in results_by_run.values():
        metrics = results.get('metrics', {})
        kl_div.append(
            metrics.get('model_behavior_preservation', {}).get('kl_div_score', 0)
        )
        ce_loss.append(
            metrics.get('model_performance_preservation', {}).get('ce_loss_score', 0)
        )
    
    # KL Divergence
    ax1.bar(runs, kl_div)
    ax1.set_title('KL Divergence Score')
    ax1.set_xticklabels([labels.get(run, run) for run in runs], rotation=45)
    
    # Cross Entropy Loss
    ax2.bar(runs, ce_loss)
    ax2.set_title('Cross Entropy Loss Score')
    ax2.set_xticklabels([labels.get(run, run) for run in runs], rotation=45)
    
    plt.tight_layout()
    plt.savefig('model_preservation.png')
    plt.close()

def main():
    # Load results from all runs
    results_by_run = {}
    for run_name in labels.keys():
        results = load_results(run_name)
        if results is not None:
            results_by_run[run_name] = results
    
    if not results_by_run:
        print("No results found!")
        return
    
    # Generate plots
    plot_reconstruction_metrics(results_by_run)
    plot_sparsity_metrics(results_by_run)
    plot_model_preservation(results_by_run)
    
    print("Plots generated successfully!")

if __name__ == "__main__":
    main()
