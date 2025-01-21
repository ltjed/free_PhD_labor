import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Style configuration
plt.style.use('seaborn-v0_8-darkgrid')  # Using a specific seaborn style
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

# Define runs to plot and their labels
labels = {
    "run_1": "Low-Rank SAE",
    "run_2": "Alternating Opt + Ortho",
    "run_3": "WMDP Integration",
    "run_4": "Gradient-Guided Isolation",
    "run_5": "Dynamic Clustering"
}

def load_run_data(run_dir):
    """Load results from a run directory."""
    try:
        with open(os.path.join(run_dir, "final_info.json"), 'r') as f:
            info = json.load(f)
        with open(os.path.join(run_dir, "all_results.npy"), 'rb') as f:
            results = np.load(f, allow_pickle=True).item()
        return info, results
    except FileNotFoundError:
        return None, None

def plot_unlearning_scores():
    """Plot unlearning scores across runs."""
    scores = []
    runs = []
    
    for run_dir, label in labels.items():
        info, _ = load_run_data(run_dir)
        if info and 'eval_result_metrics' in info:
            score = info['eval_result_metrics'].get('unlearning', {}).get('unlearning_score', 0)
            scores.append(score)
            runs.append(label)
    
    if scores:
        plt.figure(figsize=(12, 6))
        plt.bar(runs, scores)
        plt.title('Unlearning Performance Across Runs')
        plt.ylabel('Unlearning Score')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('unlearning_scores.png')
        plt.close()

def plot_training_loss():
    """Plot training loss curves for each run."""
    plt.figure(figsize=(12, 6))
    
    for run_dir, label in labels.items():
        _, results = load_run_data(run_dir)
        if results and 'training_log' in results:
            losses = [log.get('loss', 0) for log in results['training_log'] if isinstance(log, dict)]
            if losses:
                plt.plot(losses, label=label)
    
    plt.title('Training Loss Over Time')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('training_loss.png')
    plt.close()

def plot_feature_sparsity():
    """Plot feature activation sparsity across runs."""
    sparsity_values = []
    runs = []
    
    for run_dir, label in labels.items():
        _, results = load_run_data(run_dir)
        if results and 'final_info' in results:
            sparsity = results['final_info'].get('sparsity_penalty', 0)
            sparsity_values.append(sparsity)
            runs.append(label)
    
    if sparsity_values:
        plt.figure(figsize=(12, 6))
        plt.bar(runs, sparsity_values)
        plt.title('Feature Sparsity Across Runs')
        plt.ylabel('Sparsity Penalty')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('feature_sparsity.png')
        plt.close()

def main():
    """Generate all plots."""
    os.makedirs('plots', exist_ok=True)
    
    # Generate individual plots
    plot_unlearning_scores()
    plot_training_loss()
    plot_feature_sparsity()
    
    print("Plots generated successfully!")

if __name__ == "__main__":
    main()
