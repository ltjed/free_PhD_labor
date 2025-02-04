import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Dictionary mapping run names to their directories and descriptions
labels = {
    "run_0": "Baseline",
    "run_5": "Moderate Sparsity + High Orthogonality",
    "run_6": "Extended Warmup Period"
}

def load_results(run_dir):
    """Load results from a run directory."""
    with open(Path(run_dir) / "final_info.json") as f:
        return json.load(f)

def plot_sparsity_metrics():
    """Plot L0 sparsity comparison across runs."""
    l0_values = []
    run_names = []
    
    for run, label in labels.items():
        results = load_results(run)
        core_metrics = results.get("core evaluation results", {}).get("metrics", {})
        l0 = core_metrics.get("sparsity", {}).get("l0", 0)
        l0_values.append(l0)
        run_names.append(label)
    
    plt.figure(figsize=(10, 6))
    plt.bar(run_names, l0_values)
    plt.title("L0 Sparsity Comparison Across Runs")
    plt.ylabel("L0 Sparsity")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("sparsity_comparison.png")
    plt.close()

def plot_reconstruction_quality():
    """Plot reconstruction quality metrics across runs."""
    metrics = {
        "MSE": [],
        "Cosine Similarity": [],
        "Explained Variance": []
    }
    run_names = []
    
    for run, label in labels.items():
        results = load_results(run)
        recon_metrics = results.get("core evaluation results", {}).get("metrics", {}).get("reconstruction_quality", {})
        
        metrics["MSE"].append(recon_metrics.get("mse", 0))
        metrics["Cosine Similarity"].append(recon_metrics.get("cossim", 0))
        metrics["Explained Variance"].append(recon_metrics.get("explained_variance", 0))
        run_names.append(label)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, (metric_name, values) in enumerate(metrics.items()):
        axes[i].bar(run_names, values)
        axes[i].set_title(metric_name)
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig("reconstruction_quality.png")
    plt.close()

def plot_feature_separation():
    """Plot SCR metrics across different thresholds."""
    thresholds = [2, 5, 10, 20, 50, 100]
    scr_values = {}
    
    for run, label in labels.items():
        results = load_results(run)
        scr_metrics = results.get("scr and tpp evaluations results", {}).get("eval_result_metrics", {}).get("scr_metrics", {})
        values = []
        for t in thresholds:
            metric_key = f"scr_metric_threshold_{t}"
            values.append(scr_metrics.get(metric_key, 0))
        scr_values[label] = values
    
    plt.figure(figsize=(10, 6))
    for label, values in scr_values.items():
        plt.plot(thresholds, values, marker='o', label=label)
    
    plt.xlabel("Threshold")
    plt.ylabel("SCR Metric")
    plt.title("Feature Separation Across Thresholds")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("feature_separation.png")
    plt.close()

def plot_model_preservation():
    """Plot model behavior preservation metrics."""
    metrics = {
        "KL Divergence Score": [],
        "CE Loss Score": []
    }
    run_names = []
    
    for run, label in labels.items():
        results = load_results(run)
        preservation_metrics = results.get("core evaluation results", {}).get("metrics", {}).get("model_behavior_preservation", {})
        
        metrics["KL Divergence Score"].append(preservation_metrics.get("kl_div_score", 0))
        metrics["CE Loss Score"].append(preservation_metrics.get("ce_loss_score", 0))
        run_names.append(label)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, (metric_name, values) in enumerate(metrics.items()):
        axes[i].bar(run_names, values)
        axes[i].set_title(metric_name)
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig("model_preservation.png")
    plt.close()

if __name__ == "__main__":
    # Set style for all plots
    plt.style.use('default')
    # Add grid and improve visibility
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.facecolor'] = '#f0f0f0'
    
    # Generate all plots
    plot_sparsity_metrics()
    plot_reconstruction_quality()
    plot_feature_separation()
    plot_model_preservation()
