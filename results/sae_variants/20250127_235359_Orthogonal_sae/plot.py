import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Dictionary mapping run directories to their labels
labels = {
    "run_0": "Baseline",
    "run_1": "α=0.0 (Strict)",
    "run_2": "α=1.0 (Flexible)", 
    "run_3": "α=0.5 (Balanced)",
    "run_4": "α=0.25 (Strong)",
    "run_5": "α=0.125 (Very Strong)",
    "run_6": "α=0.0625 (Extreme)"
}

def load_results(run_dir):
    """Load results from a run directory."""
    results_path = Path(run_dir) / "final_info.json"
    if not results_path.exists():
        return None
    
    with open(results_path) as f:
        data = json.load(f)
    return data.get("training results for layer 19", {}).get("final_info", {})

def load_eval_metrics(run_dir):
    """Load evaluation metrics from a run directory."""
    results_path = Path(run_dir) / "final_info.json"
    if not results_path.exists():
        return None
    
    with open(results_path) as f:
        data = json.load(f)
    return data.get("core evaluation results", {}).get("metrics", {})

def plot_training_loss():
    """Plot final training loss across runs."""
    losses = []
    names = []
    
    for run_dir, label in labels.items():
        results = load_results(run_dir)
        if results and "final_loss" in results:
            losses.append(results["final_loss"])
            names.append(label)
    
    plt.figure(figsize=(10, 6))
    plt.bar(names, losses)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Final Training Loss')
    plt.title('Training Loss Comparison Across Orthogonality Constraints')
    plt.tight_layout()
    plt.savefig('training_loss_comparison.png')
    plt.close()

def plot_metrics_comparison():
    """Plot key evaluation metrics across runs."""
    kl_divs = []
    cossims = []
    l0_sparsity = []
    names = []
    
    for run_dir, label in labels.items():
        metrics = load_eval_metrics(run_dir)
        if metrics:
            kl_divs.append(metrics.get("model_behavior_preservation", {}).get("kl_div_score", 0))
            cossims.append(metrics.get("reconstruction_quality", {}).get("cossim", 0))
            l0_sparsity.append(metrics.get("sparsity", {}).get("l0", 0))
            names.append(label)
    
    # Plot KL divergence
    plt.figure(figsize=(10, 6))
    plt.plot(names, kl_divs, 'o-', label='KL Divergence Score')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('KL Divergence Score')
    plt.title('Model Behavior Preservation Across Orthogonality Constraints')
    plt.legend()
    plt.tight_layout()
    plt.savefig('kl_divergence_comparison.png')
    plt.close()
    
    # Plot cosine similarity
    plt.figure(figsize=(10, 6))
    plt.plot(names, cossims, 'o-', label='Cosine Similarity')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Cosine Similarity')
    plt.title('Reconstruction Quality Across Orthogonality Constraints')
    plt.legend()
    plt.tight_layout()
    plt.savefig('reconstruction_quality_comparison.png')
    plt.close()
    
    # Plot L0 sparsity
    plt.figure(figsize=(10, 6))
    plt.plot(names, l0_sparsity, 'o-', label='L0 Sparsity')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('L0 Sparsity')
    plt.title('Feature Activation Across Orthogonality Constraints')
    plt.legend()
    plt.tight_layout()
    plt.savefig('sparsity_comparison.png')
    plt.close()

def main():
    """Generate all plots."""
    os.makedirs('plots', exist_ok=True)
    
    # Generate individual plots
    plot_training_loss()
    plot_metrics_comparison()
    
    print("Plots have been generated in the current directory:")
    print("- training_loss_comparison.png")
    print("- kl_divergence_comparison.png") 
    print("- reconstruction_quality_comparison.png")
    print("- sparsity_comparison.png")

if __name__ == "__main__":
    main()
