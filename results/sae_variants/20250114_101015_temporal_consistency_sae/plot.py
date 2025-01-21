import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

# Define which runs to plot and their labels
labels = {
    "run_0": "Baseline",
    "run_1": "Initial Temporal Consistency",
    "run_2": "Training Init Debug",
    "run_3": "Enhanced Gradients",
    "run_4": "Training Loop Debug",
    "run_5": "Simplified Adam"
}

# Define metrics to plot
metrics = {
    'kl_div_score': 'KL Divergence Score',
    'ce_loss_score': 'Cross-Entropy Loss',
    'explained_variance': 'Explained Variance',
    'mse': 'Reconstruction MSE',
    'cossim': 'Cosine Similarity',
    'l0': 'L0 Sparsity',
    'l1': 'L1 Sparsity',
    'l2_ratio': 'L2 Norm Ratio'
}

def load_results(run_dir):
    """Load evaluation results from a run directory"""
    results_path = os.path.join(run_dir, "final_info.json")
    if not os.path.exists(results_path):
        return None
        
    with open(results_path) as f:
        return json.load(f)

def plot_metrics(metric_data, metric_name):
    """Plot a single metric across runs"""
    plt.figure(figsize=(10, 6))
    
    # Plot each run's metric value
    for run, label in labels.items():
        if run in metric_data:
            plt.bar(label, metric_data[run], alpha=0.7)
    
    plt.title(f'{metrics[metric_name]} Comparison')
    plt.ylabel(metrics[metric_name])
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{metric_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_progress(run_dirs):
    """Plot training loss progression for runs that have training logs"""
    plt.figure(figsize=(12, 6))
    
    for run, label in labels.items():
        run_dir = run_dirs[run]
        training_log_path = os.path.join(run_dir, "all_results.npy")
        
        if os.path.exists(training_log_path):
            # Load training log
            with open(training_log_path, 'rb') as f:
                results = np.load(f, allow_pickle=True).item()
            
            # Extract loss values
            if 'training_log' in results:
                steps = [x['step'] for x in results['training_log']]
                losses = [x['loss'] for x in results['training_log']]
                
                plt.plot(steps, losses, label=label, alpha=0.8)
    
    plt.title('Training Loss Progression')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/training_progress.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Find all run directories
    run_dirs = {run: run for run in labels.keys() if os.path.exists(run)}
    
    # Collect metrics across runs
    metric_data = defaultdict(dict)
    
    for run, run_dir in run_dirs.items():
        results = load_results(run_dir)
        if results and 'metrics' in results:
            for metric in metrics:
                if metric in results['metrics']:
                    metric_data[metric][run] = results['metrics'][metric]
    
    # Plot each metric
    for metric in metrics:
        if metric in metric_data:
            plot_metrics(metric_data[metric], metric)
    
    # Plot training progress
    plot_training_progress(run_dirs)

if __name__ == "__main__":
    main()
