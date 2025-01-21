import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Define which runs to plot and their labels
labels = {
    "run_0": "Baseline",
    "run_1": "Initial Temporal Consistency",
    "run_2": "Increased Window Size",
    "run_3": "Feature Correlation Analysis", 
    "run_4": "Gradient Clipping + Norm"
}

# Metrics to plot
metrics = [
    'reconstruction_quality.explained_variance',
    'reconstruction_quality.mse',
    'reconstruction_quality.cossim',
    'sparsity.l0',
    'sparsity.l1',
    'model_behavior_preservation.kl_div_score',
    'model_performance_preservation.ce_loss_score'
]

def load_run_data(run_dir):
    """Load evaluation results from a run directory"""
    results_path = os.path.join(run_dir, "all_results.npy")
    if not os.path.exists(results_path):
        return None
        
    with open(results_path, 'rb') as f:
        data = np.load(f, allow_pickle=True).item()
    
    # Get the final evaluation metrics
    if 'training_log' in data and len(data['training_log']) > 0:
        return data['training_log'][-1]['metrics']
    return None

def plot_metric_comparison(metric_data, metric_name):
    """Plot comparison of a single metric across runs"""
    plt.figure(figsize=(10, 6))
    
    # Extract run names and values
    run_names = list(metric_data.keys())
    values = [metric_data[run] for run in run_names]
    
    # Create bar plot
    bars = plt.bar(run_names, values)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.title(f'{metric_name} Comparison')
    plt.ylabel('Metric Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{metric_name.replace(".", "_")}_comparison.png')
    plt.close()

def main():
    # Load data for all runs
    all_data = {}
    for run_dir, label in labels.items():
        data = load_run_data(run_dir)
        if data is not None:
            all_data[run_dir] = data
    
    # Organize metrics for comparison
    metric_comparisons = defaultdict(dict)
    for metric in metrics:
        for run_dir, data in all_data.items():
            # Navigate nested metric structure
            parts = metric.split('.')
            value = data
            try:
                for part in parts:
                    value = value[part]
                metric_comparisons[metric][run_dir] = value
            except (KeyError, TypeError):
                metric_comparisons[metric][run_dir] = np.nan
    
    # Plot each metric comparison
    for metric, values in metric_comparisons.items():
        plot_metric_comparison(values, metric)
        
    print("Plots saved as PNG files")

if __name__ == "__main__":
    main()
