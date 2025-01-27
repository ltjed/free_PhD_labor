import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Dictionary mapping run numbers to their descriptive labels
labels = {
    "run_0": "Baseline",
    "run_5": "Progressive Orthogonality",
    "run_6": "Quadratic Scaling",
    "run_7": "Cubic Scaling",
    "run_8": "LR-Based Warmup",
    "run_9": "Activation-Based Scaling"
}

def load_results(run_dir):
    """Load results from a run directory."""
    try:
        with open(os.path.join(run_dir, "final_info.json"), 'r') as f:
            data = json.load(f)
            return data.get("training results for layer 19", {}).get("final_info", {})
    except FileNotFoundError:
        return None

def plot_loss_comparison():
    """Create bar plot comparing final losses across runs."""
    losses = []
    names = []
    
    for run, label in labels.items():
        results = load_results(run)
        if results and "final_loss" in results:
            losses.append(results["final_loss"])
            names.append(label)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(names, losses)
    plt.axhline(y=losses[0], color='r', linestyle='--', label='Baseline Loss')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.title('Final Loss Comparison Across Different Approaches')
    plt.xlabel('Training Approach')
    plt.ylabel('Final Loss')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('loss_comparison.png')
    plt.close()

def plot_relative_improvement():
    """Create plot showing relative improvement over baseline."""
    baseline = None
    improvements = []
    names = []
    
    for run, label in labels.items():
        results = load_results(run)
        if results and "final_loss" in results:
            if run == "run_0":
                baseline = results["final_loss"]
            else:
                rel_improvement = ((baseline - results["final_loss"]) / baseline) * 100
                improvements.append(rel_improvement)
                names.append(label)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(names, improvements)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    plt.title('Relative Improvement Over Baseline (%)')
    plt.xlabel('Training Approach')
    plt.ylabel('Improvement (%)')
    plt.xticks(rotation=45)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    plt.tight_layout()
    plt.savefig('relative_improvement.png')
    plt.close()

if __name__ == "__main__":
    # Create plots
    plot_loss_comparison()
    plot_relative_improvement()
