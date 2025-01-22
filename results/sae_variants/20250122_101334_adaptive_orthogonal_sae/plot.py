import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

# Define which runs to plot and their labels
labels = {
    "run_0": "Baseline",
    "run_1": "Adaptive Orthogonality (α=0.1)",
    "run_2": "Adaptive Orthogonality (α=0.3)",
    "run_3": "Dynamic Feature Grouping",
    "run_4": "Activation-Based Grouping",
    "run_5": "Cross-Feature Inhibition",
    "run_6": "Feature Importance Weighting",
    "run_7": "Task-Specific Grouping",
    "run_8": "Knowledge Graph Grouping",
    "run_9": "Multi-Mechanism Approach",
    "run_10": "Neural Inhibition + Pruning"
}

def load_results(run_dir):
    """Load evaluation results from a run directory"""
    results_path = os.path.join(run_dir, "final_info.json")
    if not os.path.exists(results_path):
        return None
        
    with open(results_path) as f:
        return json.load(f)

def plot_unlearning_scores(run_data):
    """Plot unlearning scores across all runs"""
    plt.figure(figsize=(12, 6))
    
    # Extract and plot unlearning scores
    runs = sorted(run_data.keys())
    scores = [run_data[run].get("unlearning_score", 0) for run in runs]
    
    # Create bar plot
    x = np.arange(len(runs))
    plt.bar(x, scores, color='skyblue')
    
    # Add labels and formatting
    plt.xticks(x, [labels[run] for run in runs], rotation=45, ha='right')
    plt.ylabel("Unlearning Score")
    plt.title("Unlearning Performance Across Experimental Runs")
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save and show plot
    plt.savefig("unlearning_scores.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_combined_metrics(run_data):
    """Plot multiple metrics in a combined view"""
    metrics = ['final_loss', 'sparsity_penalty', 'learning_rate']
    
    plt.figure(figsize=(12, 8))
    
    # Create subplots for each metric
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        
        # Extract metric values
        values = [run_data[run].get(metric, 0) for run in sorted(run_data.keys())]
        
        # Plot with different styles for visibility
        plt.plot(values, marker='o', linestyle='-', color='darkblue')
        plt.title(metric.replace('_', ' ').title())
        plt.xticks(range(len(values)), [labels[run] for run in sorted(run_data.keys())], 
                  rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("combined_metrics.png", dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Load results from all runs
    run_data = {}
    for run_dir in labels.keys():
        if os.path.exists(run_dir):
            results = load_results(run_dir)
            if results:
                run_data[run_dir] = results
    
    # Generate plots
    if run_data:
        plot_unlearning_scores(run_data)
        plot_combined_metrics(run_data)
    else:
        print("No valid run data found!")

if __name__ == "__main__":
    main()
