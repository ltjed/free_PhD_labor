import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import json
from collections import defaultdict
import seaborn as sns

# Configure plot style
sns.set(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12})

# Define which runs to plot and their labels
labels = {
    "run_0": "Baseline SAE",
    "run_3": "Temporal Consistency (λ=0.1)",
    "run_4": "Temporal Consistency (λ=0.5) + Gradient Clipping"
}

def load_run_data(run_dir):
    """Load training logs and final info from a run directory."""
    results_path = os.path.join(run_dir, "all_results.npy")
    final_info_path = os.path.join(run_dir, "final_info.json")
    
    if not os.path.exists(results_path):
        return None
        
    data = np.load(results_path, allow_pickle=True).item()
    with open(final_info_path) as f:
        final_info = json.load(f)
    
    return {
        "training_log": data["training_log"],
        "config": data["config"],
        "final_info": final_info
    }

def plot_training_curves(runs_data):
    """Plot training curves for multiple runs."""
    plt.figure(figsize=(12, 8))
    
    for run_name, data in runs_data.items():
        if data is None:
            continue
            
        # Extract loss components
        steps = range(len(data["training_log"]))
        total_loss = [log["loss"] for log in data["training_log"]]
        l2_loss = [log["l2_loss"] for log in data["training_log"]]
        l1_loss = [log["sparsity_loss"] for log in data["training_log"]]
        
        # Plot main loss curves
        plt.plot(steps, total_loss, label=f"{labels[run_name]} (Total)")
        plt.plot(steps, l2_loss, '--', alpha=0.5, label=f"{labels[run_name]} (L2)")
        plt.plot(steps, l1_loss, ':', alpha=0.5, label=f"{labels[run_name]} (L1)")
    
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Training Curves Across Runs")
    plt.legend()
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.close()

def plot_feature_correlations(runs_data):
    """Plot feature correlation matrices for final step of each run."""
    plt.figure(figsize=(15, 5))
    
    for i, (run_name, data) in enumerate(runs_data.items()):
        if data is None or not data["training_log"]:
            continue
            
        # Get final correlation matrix
        final_step = len(data["training_log"]) - 1
        try:
            corr_matrix = data["training_log"][final_step].get("corr_matrix", None)
            if corr_matrix is None:
                continue
                
            plt.subplot(1, len(runs_data), i+1)
            plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            plt.colorbar()
            plt.title(f"{labels[run_name]}\nFeature Correlations")
            plt.xlabel("Feature Index")
            plt.ylabel("Feature Index")
        except (IndexError, KeyError):
            continue
    
    plt.tight_layout()
    plt.savefig("feature_correlations.png")
    plt.close()

def plot_activation_distributions(runs_data):
    """Plot feature activation distributions for final step of each run."""
    plt.figure(figsize=(15, 5))
    
    for i, (run_name, data) in enumerate(runs_data.items()):
        if data is None or not data["training_log"]:
            continue
            
        # Get final activations
        final_step = len(data["training_log"]) - 1
        try:
            activations = data["training_log"][final_step].get("f", None)
            if activations is None:
                continue
                
            plt.subplot(1, len(runs_data), i+1)
            plt.hist(activations.flatten(), bins=100, log=True)
            plt.title(f"{labels[run_name]}\nActivation Distribution")
            plt.xlabel("Activation Value")
            plt.ylabel("Count (log scale)")
        except (IndexError, KeyError):
            continue
    
    plt.tight_layout()
    plt.savefig("activation_distributions.png")
    plt.close()

def main():
    # Load data for all runs
    runs_data = {run: load_run_data(run) for run in labels.keys()}
    
    # Generate plots
    plot_training_curves(runs_data)
    plot_feature_correlations(runs_data)
    plot_activation_distributions(runs_data)

if __name__ == "__main__":
    main()
