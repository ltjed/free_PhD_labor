import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path

# Dictionary mapping run numbers to descriptive labels
labels = {
    "run_1": "Initial Implementation",
    "run_2": "Improved Init & Training",
    "run_3": "Reduced Ortho Penalty",
    "run_4": "Skip Connections",
    "run_5": "Gradient Scaling",
    "run_6": "Layer Normalization",
    "run_7": "Adaptive Layer Norm",
    "run_8": "Attention Reweighting",
    "run_9": "Multi-Scale Norm"
}

# Metrics to plot
metrics = {
    "explained_variance": "Explained Variance",
    "mse": "MSE",
    "l0": "L0 Sparsity",
    "l1": "L1 Sparsity",
    "kl_div_score": "KL Divergence",
    "ce_loss_score": "CE Loss",
    "l2_ratio": "L2 Norm Ratio"
}

def load_run_data(run_dir):
    """Load metrics from a run directory."""
    try:
        with open(os.path.join(run_dir, "final_info.json"), "r") as f:
            info = json.load(f)
        
        # Load metrics from evaluation results
        metrics_file = os.path.join(run_dir, "all_results.npy")
        if os.path.exists(metrics_file):
            data = np.load(metrics_file, allow_pickle=True).item()
            if "metrics" in data:
                return {
                    "explained_variance": data["metrics"]["reconstruction_quality"]["explained_variance"],
                    "mse": data["metrics"]["reconstruction_quality"]["mse"],
                    "l0": data["metrics"]["sparsity"]["l0"],
                    "l1": data["metrics"]["sparsity"]["l1"],
                    "kl_div_score": data["metrics"]["model_behavior_preservation"]["kl_div_score"],
                    "ce_loss_score": data["metrics"]["model_performance_preservation"]["ce_loss_score"],
                    "l2_ratio": data["metrics"]["shrinkage"]["l2_ratio"]
                }
    except Exception as e:
        print(f"Error loading data from {run_dir}: {e}")
    return None

def plot_metrics():
    """Generate plots for all metrics across runs."""
    # Create output directory
    os.makedirs("plots", exist_ok=True)
    
    # Collect data from all runs
    data = {}
    for run_name in labels.keys():
        run_data = load_run_data(run_name)
        if run_data:
            data[run_name] = run_data
    
    if not data:
        print("No valid run data found!")
        return
    
    # Create plots
    for metric_key, metric_name in metrics.items():
        plt.figure(figsize=(12, 6))
        
        # Extract values and run numbers
        runs = list(data.keys())
        values = [data[run][metric_key] for run in runs]
        x = range(len(runs))
        
        # Plot bars
        plt.bar(x, values)
        
        # Customize plot
        plt.title(f"Evolution of {metric_name} Across Runs")
        plt.xlabel("Run")
        plt.ylabel(metric_name)
        plt.xticks(x, [labels[run] for run in runs], rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on top of bars
        for i, v in enumerate(values):
            plt.text(i, v, f'{v:.3f}', ha='center', va='bottom')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f"plots/{metric_key}_evolution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create combined plot for key metrics
    plt.figure(figsize=(15, 8))
    key_metrics = ["explained_variance", "l0", "kl_div_score"]
    
    for i, metric in enumerate(key_metrics):
        plt.subplot(1, 3, i+1)
        values = [data[run][metric] for run in runs]
        plt.plot(range(1, len(runs)+1), values, 'o-')
        plt.title(metrics[metric])
        plt.xlabel("Run Number")
        plt.grid(True, alpha=0.3)
        plt.xticks(range(1, len(runs)+1))
    
    plt.tight_layout()
    plt.savefig("plots/key_metrics_combined.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_metrics()
