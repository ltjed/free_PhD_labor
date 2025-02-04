import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Dictionary mapping run names to their labels in plots
labels = {
    "run_0": "Baseline (Standard SAE)",
    "run_1": "Initial Orthogonality (w=0.01)", 
    "run_2": "Increased Orthogonality (w=0.1)",
    "run_3": "Balanced Orthogonality (w=0.05)",
    "run_4": "Optimal Dictionary Size (18432)",
    "run_5": "Larger Dictionary (32768)",
    "run_6": "Strong Orthogonality (w=0.1, d=18432)"
}

def load_results(run_dir):
    """Load results from a run directory"""
    with open(Path(run_dir) / "final_info.json") as f:
        data = json.load(f)
        # Get the results for layer 12 which contains all evaluation results
        return data["training results for layer 12"]

def plot_absorption_comparison():
    """Plot absorption score comparisons across runs"""
    scores = []
    names = []
    
    for run, label in labels.items():
        results = load_results(run)
        if "absorption evaluation results" in results:
            score = results["absorption evaluation results"]["eval_result_metrics"]["mean"]["mean_absorption_score"]
            scores.append(score)
            names.append(label)
    
    plt.figure(figsize=(12, 6))
    plt.bar(names, scores)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Mean Absorption Score')
    plt.title('Absorption Score Comparison Across Runs')
    plt.tight_layout()
    plt.savefig('absorption_comparison.png')
    plt.close()

def plot_scr_metrics():
    """Plot SCR metrics across runs"""
    scr_2 = []
    scr_20 = []
    names = []
    
    for run, label in labels.items():
        results = load_results(run)
        if "scr and tpp evaluations results" in results:
            metrics = results["scr and tpp evaluations results"]["eval_result_metrics"]["scr_metrics"]
            scr_2.append(metrics["scr_dir1_threshold_2"])
            scr_20.append(metrics["scr_dir1_threshold_20"])
            names.append(label)
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(names))
    width = 0.35
    
    plt.bar(x - width/2, scr_2, width, label='SCR k=2')
    plt.bar(x + width/2, scr_20, width, label='SCR k=20')
    plt.xticks(x, names, rotation=45, ha='right')
    plt.ylabel('SCR Score')
    plt.title('SCR Metrics Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig('scr_comparison.png')
    plt.close()

def plot_reconstruction_quality():
    """Plot reconstruction quality metrics"""
    mse = []
    cossim = []
    names = []
    
    for run, label in labels.items():
        results = load_results(run)
        if "core evaluation results" in results:
            metrics = results["core evaluation results"]["metrics"]["reconstruction_quality"]
            mse.append(metrics["mse"])
            cossim.append(metrics["cossim"])
            names.append(label)
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(names))
    width = 0.35
    
    plt.bar(x - width/2, mse, width, label='MSE')
    plt.bar(x + width/2, cossim, width, label='Cosine Similarity')
    plt.xticks(x, names, rotation=45, ha='right')
    plt.ylabel('Score')
    plt.title('Reconstruction Quality Metrics')
    plt.legend()
    plt.tight_layout()
    plt.savefig('reconstruction_quality.png')
    plt.close()

def plot_sparse_probing():
    """Plot sparse probing accuracy"""
    top_1 = []
    top_20 = []
    names = []
    
    for run, label in labels.items():
        results = load_results(run)
        if "sparse probing evaluation results" in results:
            metrics = results["sparse probing evaluation results"]["eval_result_metrics"]["sae"]
            top_1.append(metrics["sae_top_1_test_accuracy"])
            top_20.append(metrics["sae_top_20_test_accuracy"])
            names.append(label)
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(names))
    width = 0.35
    
    plt.bar(x - width/2, top_1, width, label='Top-1 Accuracy')
    plt.bar(x + width/2, top_20, width, label='Top-20 Accuracy')
    plt.xticks(x, names, rotation=45, ha='right')
    plt.ylabel('Accuracy')
    plt.title('Sparse Probing Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('sparse_probing.png')
    plt.close()

if __name__ == "__main__":
    # Set style
    plt.style.use('default')  # Use default matplotlib style
    
    # Configure plot style
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    
    # Generate all plots
    plot_absorption_comparison()
    plot_scr_metrics()
    plot_reconstruction_quality() 
    plot_sparse_probing()
