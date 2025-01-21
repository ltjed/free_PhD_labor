import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

# Define which runs to plot and their labels
labels = {
    "run_0": "Baseline SAE",
    "run_4": "TemporalSAE (No Temp Loss)",
    "run_5": "TemporalSAE (Full)"
}

# Metrics to plot
metrics = {
    'reconstruction_quality': ['explained_variance', 'mse', 'cossim'],
    'sparsity': ['l0', 'l1'],
    'model_performance_preservation': ['ce_loss_score'],
    'shrinkage': ['l2_ratio']
}

def load_results(run_dir):
    """Load results from a run directory"""
    results_path = os.path.join(run_dir, "final_info.json")
    if not os.path.exists(results_path):
        return None
        
    with open(results_path) as f:
        return json.load(f)

def plot_metric_comparison(all_results, metric_category, metric_name):
    """Plot comparison of a specific metric across runs"""
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(labels))
    width = 0.6
    
    values = []
    for run_id, label in labels.items():
        results = all_results[run_id]
        if results and metric_category in results and metric_name in results[metric_category]:
            values.append(results[metric_category][metric_name])
        else:
            values.append(np.nan)
    
    bars = plt.bar(x, values, width)
    plt.xticks(x, labels.values(), rotation=45, ha='right')
    plt.ylabel(metric_name.replace('_', ' ').title())
    plt.title(f"{metric_name.replace('_', ' ').title()} Comparison")
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{metric_name}_comparison.png")
    plt.close()

def plot_training_curves(run_dirs):
    """Plot training curves for each run"""
    plt.figure(figsize=(12, 8))
    
    for run_id, label in labels.items():
        training_log_path = os.path.join(run_id, "all_results.npy")
        if not os.path.exists(training_log_path):
            continue
            
        data = np.load(training_log_path, allow_pickle=True).item()
        if 'training_log' not in data:
            continue
            
        losses = [step['loss'] for step in data['training_log'] if 'loss' in step]
        plt.plot(losses, label=label)
    
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.close()

def main():
    # Load results from all runs
    all_results = {}
    for run_id in labels:
        all_results[run_id] = load_results(run_id)
    
    # Plot individual metric comparisons
    for metric_category, metric_names in metrics.items():
        for metric_name in metric_names:
            plot_metric_comparison(all_results, metric_category, metric_name)
    
    # Plot training curves
    plot_training_curves(labels.keys())
    
    print("Plots saved to current directory")

if __name__ == "__main__":
    main()
