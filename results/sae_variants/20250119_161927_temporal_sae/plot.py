import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

# Dictionary mapping run names to their display labels
labels = {
    "run_0": "Baseline SAE",
    "run_1": "Multi-Scale TSAE",
    "run_2": "TSAE w/o Feature Sep.",
    "run_3": "Single-Scale TSAE", 
    "run_4": "Adaptive Feature SAE",
    "run_5": "Hierarchical SAE"
}

def load_results(run_dir):
    """Load training results from a run directory."""
    results_path = os.path.join(run_dir, "all_results.npy")
    if not os.path.exists(results_path):
        return None
    
    with open(results_path, 'rb') as f:
        return np.load(f, allow_pickle=True).item()

def plot_training_curves(results_dict):
    """Plot training loss curves for each run."""
    plt.figure(figsize=(10, 6))
    
    for run_name, results in results_dict.items():
        if run_name not in labels or results is None:
            continue
            
        training_log = results["training_log"]
        losses = [log["loss"] for log in training_log]
        steps = range(len(losses))
        
        plt.plot(steps, losses, label=labels[run_name])
    
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_curves.png")
    plt.close()

def plot_feature_sparsity(results_dict):
    """Plot feature activation sparsity for each run."""
    plt.figure(figsize=(10, 6))
    
    x_positions = np.arange(len(labels))
    sparsities = []
    
    for run_name in labels.keys():
        if run_name in results_dict and results_dict[run_name] is not None:
            results = results_dict[run_name]
            if "final_info" in results:
                # Calculate average sparsity from final training batch
                sparsity = results["final_info"].get("sparsity", 0)
                sparsities.append(sparsity)
            else:
                sparsities.append(0)
        else:
            sparsities.append(0)
    
    plt.bar(x_positions, sparsities, align='center')
    plt.xticks(x_positions, [labels[run] for run in labels.keys()], rotation=45)
    plt.ylabel("Feature Sparsity")
    plt.title("Feature Activation Sparsity by Model")
    plt.tight_layout()
    plt.savefig("feature_sparsity.png")
    plt.close()

def plot_unlearning_performance(results_dict):
    """Plot unlearning performance metrics."""
    plt.figure(figsize=(10, 6))
    
    x_positions = np.arange(len(labels))
    scores = []
    
    for run_name in labels.keys():
        if run_name in results_dict and results_dict[run_name] is not None:
            results = results_dict[run_name]
            if "final_info" in results:
                # Get unlearning score from evaluation results
                score = results["final_info"].get("unlearning_score", 0)
                scores.append(score)
            else:
                scores.append(0)
        else:
            scores.append(0)
    
    plt.bar(x_positions, scores, align='center')
    plt.xticks(x_positions, [labels[run] for run in labels.keys()], rotation=45)
    plt.ylabel("Unlearning Score")
    plt.title("Unlearning Performance by Model")
    plt.tight_layout()
    plt.savefig("unlearning_performance.png")
    plt.close()

def main():
    # Load results from each run
    results_dict = {}
    for run_name in labels.keys():
        results = load_results(run_name)
        if results is not None:
            results_dict[run_name] = results
    
    # Generate plots
    plot_training_curves(results_dict)
    plot_feature_sparsity(results_dict)
    plot_unlearning_performance(results_dict)

if __name__ == "__main__":
    # Set style
    plt.style.use('default')
    # Custom style settings
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 2,
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10
    })
    # Set color palette
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#99CCFF']
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)
    
    main()
