import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Define which runs to plot and their labels
labels = {
    "run_0": "Baseline SAE",
    "run_1": "Initial Temporal SAE",
    "run_2": "Conservative Temporal Weighting",
    "run_3": "Temporal Buffer Warmup",
    "run_4": "Progressive Temporal Training",
    "run_5": "Reconstruction-Focused Warmup"
}

def load_run_data(run_dir):
    """Load training data from a run directory"""
    results_path = os.path.join(run_dir, "all_results.npy")
    if not os.path.exists(results_path):
        return None
        
    data = np.load(results_path, allow_pickle=True).item()
    return data

def plot_training_metrics(run_data):
    """Plot key training metrics across runs"""
    plt.figure(figsize=(15, 10))
    
    # Plot reconstruction loss
    plt.subplot(2, 2, 1)
    for run, label in labels.items():
        data = run_data[run]
        if data and "training_log" in data:
            losses = [step["loss"] for step in data["training_log"] if "loss" in step]
            plt.plot(losses, label=label)
    plt.title("Training Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot temporal consistency
    plt.subplot(2, 2, 2)
    for run, label in labels.items():
        data = run_data[run]
        if data and "training_log" in data:
            temp_loss = [step.get("temporal_loss", 0) for step in data["training_log"]]
            plt.plot(temp_loss, label=label)
    plt.title("Temporal Consistency Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot feature activation variance
    plt.subplot(2, 2, 3)
    for run, label in labels.items():
        data = run_data[run]
        if data and "training_log" in data:
            var = [step.get("feature_variance", 0) for step in data["training_log"]]
            plt.plot(var, label=label)
    plt.title("Feature Activation Variance")
    plt.xlabel("Steps")
    plt.ylabel("Variance")
    plt.legend()
    plt.grid(True)

    # Plot reconstruction quality
    plt.subplot(2, 2, 4)
    for run, label in labels.items():
        data = run_data[run]
        if data and "final_info" in data:
            recon = data["final_info"].get("reconstruction_quality", 0)
            plt.bar(label, recon, alpha=0.6)
    plt.title("Final Reconstruction Quality")
    plt.ylabel("Explained Variance")
    plt.xticks(rotation=45)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.close()

def plot_feature_analysis(run_data):
    """Plot feature activation patterns"""
    plt.figure(figsize=(15, 5))
    
    # Plot feature lifetime
    plt.subplot(1, 2, 1)
    for run, label in labels.items():
        data = run_data[run]
        if data and "training_log" in data:
            lifetimes = [step.get("feature_lifetime", 0) for step in data["training_log"]]
            plt.plot(lifetimes, label=label)
    plt.title("Feature Lifetime")
    plt.xlabel("Steps")
    plt.ylabel("Average Lifetime")
    plt.legend()
    plt.grid(True)

    # Plot feature sparsity
    plt.subplot(1, 2, 2)
    for run, label in labels.items():
        data = run_data[run]
        if data and "training_log" in data:
            sparsity = [step.get("sparsity", 0) for step in data["training_log"]]
            plt.plot(sparsity, label=label)
    plt.title("Feature Sparsity")
    plt.xlabel("Steps")
    plt.ylabel("L0 Norm")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("feature_analysis.png")
    plt.close()

def main():
    # Load data from all runs
    run_data = {}
    for run in labels:
        data = load_run_data(run)
        if data:
            run_data[run] = data

    # Generate plots
    plot_training_metrics(run_data)
    plot_feature_analysis(run_data)

if __name__ == "__main__":
    main()
