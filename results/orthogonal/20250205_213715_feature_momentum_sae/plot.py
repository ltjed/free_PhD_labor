import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Dictionary mapping run names to their display labels
labels = {
    "run_0": "Baseline",
    "run_1": "Initial Momentum (β=0.9, τ=0.5)",
    "run_2": "High Momentum (β=0.99, τ=0.3)", 
    "run_3": "Aggressive Competition (β=0.8, τ=0.7)",
    "run_4": "Balanced High Penalty (β=0.95, τ=0.4)",
    "run_5": "Final Balanced (β=0.95, τ=0.4, λ=0.3)"
}

def load_results(run_dir):
    """Load results from a run directory"""
    with open(os.path.join(run_dir, "final_info.json"), "r") as f:
        return json.load(f)

def plot_absorption_scores():
    """Plot absorption scores across runs"""
    scores = []
    names = []
    
    for run, label in labels.items():
        try:
            results = load_results(run)
            score = results.get("absorption evaluation results", {}).get("eval_result_metrics", {}).get("mean", {}).get("mean_absorption_score", None)
            if score is not None:
                scores.append(score)
                names.append(label)
        except:
            continue
    
    plt.figure(figsize=(12, 6))
    plt.bar(names, scores)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Absorption Score (lower is better)')
    plt.title('Feature Absorption Scores Across Runs')
    plt.tight_layout()
    plt.savefig('absorption_comparison.png')
    plt.close()

def plot_reconstruction_quality():
    """Plot reconstruction quality metrics"""
    explained_var = []
    mse = []
    names = []
    
    for run, label in labels.items():
        try:
            results = load_results(run)
            metrics = results.get("core evaluation results", {}).get("metrics", {}).get("reconstruction_quality", {})
            if metrics:
                explained_var.append(metrics.get("explained_variance", 0))
                mse.append(metrics.get("mse", 0))
                names.append(label)
        except:
            continue
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(names))
    width = 0.35
    
    plt.bar(x - width/2, explained_var, width, label='Explained Variance')
    plt.bar(x + width/2, mse, width, label='MSE')
    plt.xticks(x, names, rotation=45, ha='right')
    plt.ylabel('Score')
    plt.title('Reconstruction Quality Metrics')
    plt.legend()
    plt.tight_layout()
    plt.savefig('reconstruction_quality.png')
    plt.close()

def plot_sparsity_metrics():
    """Plot sparsity metrics"""
    l0_sparsity = []
    l1_sparsity = []
    names = []
    
    for run, label in labels.items():
        try:
            results = load_results(run)
            metrics = results.get("core evaluation results", {}).get("metrics", {}).get("sparsity", {})
            if metrics:
                l0_sparsity.append(metrics.get("l0", 0))
                l1_sparsity.append(metrics.get("l1", 0))
                names.append(label)
        except:
            continue
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(names))
    width = 0.35
    
    plt.bar(x - width/2, l0_sparsity, width, label='L0 Sparsity')
    plt.bar(x + width/2, l1_sparsity, width, label='L1 Sparsity')
    plt.xticks(x, names, rotation=45, ha='right')
    plt.ylabel('Sparsity Value')
    plt.title('Sparsity Metrics Across Runs')
    plt.legend()
    plt.tight_layout()
    plt.savefig('sparsity_metrics.png')
    plt.close()

def plot_sparse_probing_accuracy():
    """Plot sparse probing accuracy"""
    accuracies = []
    names = []
    
    for run, label in labels.items():
        try:
            results = load_results(run)
            acc = results.get("sparse probing evaluation results", {}).get("eval_result_metrics", {}).get("sae", {}).get("sae_test_accuracy", None)
            if acc is not None:
                accuracies.append(acc)
                names.append(label)
        except:
            continue
    
    plt.figure(figsize=(12, 6))
    plt.bar(names, accuracies)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Test Accuracy')
    plt.title('Sparse Probing Accuracy Across Runs')
    plt.tight_layout()
    plt.savefig('sparse_probing_accuracy.png')
    plt.close()

if __name__ == "__main__":
    # Create all plots
    plot_absorption_scores()
    plot_reconstruction_quality()
    plot_sparsity_metrics()
    plot_sparse_probing_accuracy()
    
    print("Plots have been generated:")
    print("- absorption_comparison.png")
    print("- reconstruction_quality.png") 
    print("- sparsity_metrics.png")
    print("- sparse_probing_accuracy.png")
