import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

# Define which runs to plot and their labels
labels = {
    "run_0": "Baseline",
    "run_1": "Adaptive Ortho (α=0.1)",
    "run_2": "Adaptive Ortho (α=0.05)",
    "run_3": "Batch Feature Grouping",
    "run_4": "Task-Specific Masking",
    "run_5": "Dynamic α Scheduling",
    "run_6": "Feature Importance Ortho",
    "run_7": "Hierarchical Grouping",
    "run_8": "Cross-Task Inhibition",
    "run_9": "Gradient Disentanglement",
    "run_10": "Subspace Clustering"
}

# Metrics to plot
METRICS = [
    'unlearning_score',
    'l2_loss',
    'sparsity_loss',
    'loss'
]

def load_results(run_dir):
    """Load results from a run directory"""
    results_path = os.path.join(run_dir, "all_results.npy")
    if not os.path.exists(results_path):
        return None
        
    results = np.load(results_path, allow_pickle=True).item()
    return results

def plot_metrics(runs, output_dir="plots"):
    """Plot metrics across runs"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Aggregate metrics across runs
    metrics_data = defaultdict(dict)
    for run_name, run_label in labels.items():
        results = load_results(run_name)
        if results is None:
            continue
            
        # Get training log
        training_log = results.get('training_log', [])
        if not training_log:
            continue
            
        # Get final metrics
        final_info = results.get('final_info', {})
        eval_results = final_info.get('eval_result_metrics', {})
        
        # Store metrics
        for metric in METRICS:
            if metric in eval_results.get('unlearning', {}):
                metrics_data[metric][run_label] = eval_results['unlearning'][metric]
            elif metric in final_info:
                metrics_data[metric][run_label] = final_info[metric]
            else:
                # Try to get from last training step
                last_step = training_log[-1]['losses']
                if metric in last_step:
                    metrics_data[metric][run_label] = last_step[metric]

    # Plot each metric
    for metric, data in metrics_data.items():
        plt.figure(figsize=(10, 6))
        
        # Sort by run order in labels dict
        sorted_labels = [l for l in labels.values() if l in data]
        values = [data[l] for l in sorted_labels]
        
        # Create bar plot
        bars = plt.bar(sorted_labels, values)
        plt.title(f"{metric.replace('_', ' ').title()} Comparison")
        plt.ylabel(metric.replace('_', ' ').title())
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}.png"))
        plt.close()

def plot_training_curves(runs, output_dir="plots"):
    """Plot training curves for each run"""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    for run_name, run_label in labels.items():
        results = load_results(run_name)
        if results is None:
            continue
            
        training_log = results.get('training_log', [])
        if not training_log:
            continue
            
        # Get loss values
        steps = range(len(training_log))
        losses = [step['losses']['loss'] for step in training_log]
        
        plt.plot(steps, losses, label=run_label)
    
    plt.title("Training Loss Curves")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"))
    plt.close()

if __name__ == "__main__":
    # Create plots
    plot_metrics(labels)
    plot_training_curves(labels)
    
    print(f"Plots saved to 'plots' directory")
