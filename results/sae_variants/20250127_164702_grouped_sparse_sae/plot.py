import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
import json
import os

# Dictionary mapping run names to their labels in plots
labels = {
    "Run 0": "Baseline (No Groups)",
    "Run 1": "3 Groups (γ=2.0)",
    "Run 2": "5 Groups (γ=2.0)", 
    "Run 3": "5 Groups (γ=1.5)",
    "Run 4": "5 Groups (γ=1.5, β=0.06)"
}

def load_results(results_dir):
    """Load results from the final_info.json file"""
    results_path = os.path.join(results_dir, "final_info.json")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"No results file found at {results_path}")
    
    with open(results_path, 'r') as f:
        return json.load(f)

def plot_metrics_comparison():
    """Plot comparison of key metrics across runs"""
    # Metrics from the notes
    runs = list(labels.keys())
    metrics = {
        'Training Loss': [200.23, 85.87, 110.09, 90.55, 105.89],
        'KL Divergence': [2.06, 0.047, 0.155, 0.056, 0.99],
        'Explained Variance': [0.31, 0.926, 0.820, 0.910, 0.863],
        'L0 Sparsity': [85, 1711, 1206, 1634, 1379]
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Comparison of Key Metrics Across Runs', fontsize=16)
    
    for (metric, values), ax in zip(metrics.items(), axes.flat):
        ax.bar(range(len(runs)), values)
        ax.set_xticks(range(len(runs)))
        ax.set_xticklabels([labels[run] for run in runs], rotation=45, ha='right')
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_group_activations(features, group_sizes, title="Group Activations"):
    """Plot activation statistics per group"""
    num_groups = len(group_sizes)
    group_stats = []
    start_idx = 0
    
    for i, size in enumerate(group_sizes):
        end_idx = start_idx + size
        group_features = features[:, start_idx:end_idx]
        
        # Calculate statistics
        sparsity = (group_features > 0).float().mean().item()
        mean_activation = group_features[group_features > 0].mean().item() if group_features.max() > 0 else 0
        
        group_stats.append({
            'Group': i+1,
            'Sparsity': sparsity,
            'Mean Activation': mean_activation
        })
        start_idx = end_idx
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot sparsity
    groups = [stat['Group'] for stat in group_stats]
    sparsities = [stat['Sparsity'] for stat in group_stats]
    ax1.bar(groups, sparsities)
    ax1.set_xlabel('Group')
    ax1.set_ylabel('Activation Rate')
    ax1.set_title('Group Sparsity')
    
    # Plot mean activations
    means = [stat['Mean Activation'] for stat in group_stats]
    ax2.bar(groups, means)
    ax2.set_xlabel('Group')
    ax2.set_ylabel('Mean Activation')
    ax2.set_title('Group Mean Activation')
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig

def plot_feature_usage(features, group_sizes, title="Feature Usage Pattern"):
    """Plot heatmap of feature activations across groups"""
    plt.figure(figsize=(10, 6))
    sns.heatmap(features.T > 0, cmap='viridis', cbar_kws={'label': 'Active'})
    
    # Add group boundary lines
    cumsum = 0
    for size in group_sizes[:-1]:
        cumsum += size
        plt.axhline(y=cumsum, color='r', linestyle='-')
    
    plt.title(title)
    plt.xlabel('Sample')
    plt.ylabel('Feature')
    return plt.gcf()

def plot_penalty_progression():
    """Plot the penalty progression across groups for different runs"""
    runs = {
        "Run 1": {"gamma": 2.0, "base": 0.04, "groups": 3},
        "Run 2": {"gamma": 2.0, "base": 0.04, "groups": 5},
        "Run 3": {"gamma": 1.5, "base": 0.04, "groups": 5},
        "Run 4": {"gamma": 1.5, "base": 0.06, "groups": 5}
    }
    
    plt.figure(figsize=(10, 6))
    for run, params in runs.items():
        penalties = [params["base"] * (params["gamma"] ** i) for i in range(params["groups"])]
        plt.plot(range(1, params["groups"] + 1), penalties, marker='o', label=labels[run])
    
    plt.xlabel('Group Number')
    plt.ylabel('L1 Penalty')
    plt.title('Penalty Progression Across Groups')
    plt.legend()
    plt.grid(True, alpha=0.3)
    return plt.gcf()

if __name__ == "__main__":
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    
    # Generate and save all plots
    metrics_fig = plot_metrics_comparison()
    metrics_fig.savefig("plots/metrics_comparison.png", bbox_inches='tight', dpi=300)
    
    penalty_fig = plot_penalty_progression()
    penalty_fig.savefig("plots/penalty_progression.png", bbox_inches='tight', dpi=300)
    
    plt.close('all')
    print("Plots have been saved to the 'plots' directory")
