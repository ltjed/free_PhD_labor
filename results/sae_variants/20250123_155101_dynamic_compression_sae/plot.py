import matplotlib.pyplot as plt
import numpy as np
import os
import json
from pathlib import Path

# Map run numbers to descriptive labels
labels = {
    "run_1": "Initial Dynamic",
    "run_2": "Gradual+Warmup", 
    "run_3": "Enhanced Gradual",
    "run_4": "Adaptive Features",
    "run_5": "Contrastive",
    "run_6": "Hierarchical",
    "run_7": "Parallel Paths",
    "run_8": "Minimal Gating",
    "run_9": "Additive Mix"
}

def load_run_results(run_dir):
    """Load results from a run directory."""
    results_path = os.path.join(run_dir, "all_results.npy")
    if os.path.exists(results_path):
        return np.load(results_path, allow_pickle=True).item()
    return None

def plot_metrics_comparison(base_dir):
    """Plot comparison of key metrics across runs."""
    metrics = {
        'explained_variance': [],
        'mse': [],
        'l0_sparsity': [],
        'kl_div': []
    }
    
    run_labels = []
    
    # Collect metrics from each run
    for run_name in sorted(labels.keys()):
        run_dir = os.path.join(base_dir, run_name)
        results = load_run_results(run_dir)
        
        if results and 'metrics' in results:
            m = results['metrics']
            metrics['explained_variance'].append(m['reconstruction_quality']['explained_variance'])
            metrics['mse'].append(m['reconstruction_quality']['mse'])
            metrics['l0_sparsity'].append(m['sparsity']['l0'])
            metrics['kl_div'].append(m['model_behavior_preservation']['kl_div_with_sae'])
            run_labels.append(labels[run_name])
    
    # Create comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Compression Performance Metrics Across Runs', fontsize=16)
    
    # Explained Variance
    ax1.bar(run_labels, metrics['explained_variance'])
    ax1.set_title('Explained Variance')
    ax1.set_xticklabels(run_labels, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # MSE
    ax2.bar(run_labels, metrics['mse'])
    ax2.set_title('Reconstruction MSE')
    ax2.set_xticklabels(run_labels, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # L0 Sparsity
    ax3.bar(run_labels, metrics['l0_sparsity'])
    ax3.set_title('L0 Sparsity')
    ax3.set_xticklabels(run_labels, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # KL Divergence
    ax4.bar(run_labels, metrics['kl_div'])
    ax4.set_title('KL Divergence')
    ax4.set_xticklabels(run_labels, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('compression_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_curves(base_dir):
    """Plot training curves across runs."""
    plt.figure(figsize=(12, 6))
    
    for run_name in sorted(labels.keys()):
        run_dir = os.path.join(base_dir, run_name)
        results = load_run_results(run_dir)
        
        if results and 'training_log' in results:
            losses = [log.get('loss', float('nan')) for log in results['training_log']]
            steps = range(len(losses))
            plt.plot(steps, losses, label=labels[run_name], alpha=0.7)
    
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Curves')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    base_dir = "."  # Assumes running from project root
    plot_metrics_comparison(base_dir)
    plot_training_curves(base_dir)
