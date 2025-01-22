import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Dictionary mapping run numbers to their descriptive labels
labels = {
    "run_0": "Baseline SAE",
    "run_1": "Fixed Orthogonal",
    "run_2": "Adaptive Orthogonal (32 groups)",
    "run_3": "Enhanced Adaptive (64 groups)",
    "run_4": "Hierarchical Static",
    "run_5": "Dynamic Hierarchical",
    "run_6": "Contrastive Hierarchical"
}

def load_run_data(run_dir):
    """Load results from a run directory."""
    info_path = os.path.join(run_dir, "final_info.json")
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            return json.load(f)
    return None

def plot_unlearning_scores():
    """Plot unlearning scores across runs."""
    scores = []
    run_labels = []
    
    for run_name, label in labels.items():
        data = load_run_data(run_name)
        if data and 'eval_result_metrics' in data:
            score = data['eval_result_metrics'].get('unlearning', {}).get('unlearning_score', 0)
            scores.append(score)
            run_labels.append(label)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=run_labels, y=scores)
    plt.title("Unlearning Scores Across Different SAE Variants")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Unlearning Score")
    plt.tight_layout()
    plt.savefig("unlearning_scores.png")
    plt.close()

def plot_loss_comparison():
    """Plot final loss values across runs."""
    losses = []
    run_labels = []
    
    for run_name, label in labels.items():
        data = load_run_data(run_name)
        if data and 'final_loss' in data:
            loss = data['final_loss']
            if loss is not None:
                losses.append(loss)
                run_labels.append(label)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=run_labels, y=losses)
    plt.title("Final Loss Values Across Different SAE Variants")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Final Loss")
    plt.tight_layout()
    plt.savefig("final_losses.png")
    plt.close()

def plot_architecture_comparison():
    """Create a visual comparison of architectural differences."""
    architectures = {
        "Baseline SAE": ["Standard Features"],
        "Fixed Orthogonal": ["Orthogonal Features", "Fixed α=0.1"],
        "Adaptive Orthogonal": ["32 Groups", "Dynamic α", "Group Updates"],
        "Enhanced Adaptive": ["64 Groups", "Higher α", "Frequent Updates"],
        "Hierarchical Static": ["8 Coarse Groups", "8 Fine Groups", "Fixed Structure"],
        "Dynamic Hierarchical": ["Attention", "Learned Groups", "Entropy Reg."],
        "Contrastive Hierarchical": ["Contrastive Loss", "Dynamic Groups", "InfoNCE"]
    }
    
    # Create a matrix representation
    max_features = max(len(v) for v in architectures.values())
    matrix = np.zeros((len(architectures), max_features))
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(matrix, 
                xticklabels=range(1, max_features + 1),
                yticklabels=list(architectures.keys()),
                cmap='YlOrRd',
                cbar=False)
    
    # Add text annotations
    for i, (arch, features) in enumerate(architectures.items()):
        for j, feature in enumerate(features):
            plt.text(j + 0.5, i + 0.5, feature,
                    ha='center', va='center')
    
    plt.title("Architectural Features Across SAE Variants")
    plt.xlabel("Feature Number")
    plt.tight_layout()
    plt.savefig("architecture_comparison.png")
    plt.close()

def main():
    """Generate all plots."""
    # Set basic style
    plt.rcParams['figure.figsize'] = [12, 6]
    plt.rcParams['axes.grid'] = True
    
    # Create plots
    plot_unlearning_scores()
    plot_loss_comparison()
    plot_architecture_comparison()
    
    print("Plots generated successfully!")

if __name__ == "__main__":
    main()
