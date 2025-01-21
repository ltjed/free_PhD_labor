import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Dictionary mapping run numbers to their descriptive labels
labels = {
    "run_0": "Baseline SAE",
    "run_1": "Basic Contrastive",
    "run_2": "Enhanced Contrastive",
    "run_3": "Adaptive Selection",
    "run_4": "Hierarchical + MI"
}

def load_run_data(run_dir):
    """Load results from a run directory."""
    try:
        with open(os.path.join(run_dir, "final_info.json"), "r") as f:
            final_info = json.load(f)
        with open(os.path.join(run_dir, "all_results.npy"), "rb") as f:
            results = np.load(f, allow_pickle=True).item()
        return final_info, results
    except FileNotFoundError:
        return None, None

def plot_unlearning_scores(runs_data):
    """Plot unlearning scores across runs."""
    plt.figure(figsize=(10, 6))
    scores = []
    run_labels = []
    
    for run_name, label in labels.items():
        final_info, _ = runs_data[run_name]
        if final_info and 'eval_result_metrics' in final_info:
            score = final_info['eval_result_metrics'].get('unlearning', {}).get('unlearning_score', 0)
            scores.append(score)
            run_labels.append(label)
    
    plt.bar(run_labels, scores)
    plt.title("Unlearning Scores Across Runs")
    plt.ylabel("Unlearning Score")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("unlearning_scores.png")
    plt.close()

def plot_training_losses(runs_data):
    """Plot training losses across runs."""
    plt.figure(figsize=(12, 6))
    
    for run_name, label in labels.items():
        _, results = runs_data[run_name]
        if results and 'training_log' in results:
            losses = [log.get('loss', None) for log in results['training_log'] if isinstance(log, dict)]
            if losses:
                plt.plot(losses, label=label, alpha=0.7)
    
    plt.title("Training Loss Progression")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_losses.png")
    plt.close()

def plot_feature_stats(runs_data):
    """Plot feature importance statistics."""
    plt.figure(figsize=(10, 6))
    stats = []
    run_labels = []
    
    for run_name, label in labels.items():
        _, results = runs_data[run_name]
        if results and 'config' in results:
            config = results['config']
            if 'feature_importance_threshold' in config:
                stats.append(config['feature_importance_threshold'])
                run_labels.append(label)
    
    if stats:
        plt.bar(run_labels, stats)
        plt.title("Feature Importance Thresholds")
        plt.ylabel("Threshold Value")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig("feature_thresholds.png")
    plt.close()

def main():
    # Set style
    plt.style.use('default')
    # Use a colorful default color cycle
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    
    # Load data from all runs
    runs_data = {}
    for run_name in labels.keys():
        final_info, results = load_run_data(run_name)
        runs_data[run_name] = (final_info, results)
    
    # Generate plots
    plot_unlearning_scores(runs_data)
    plot_training_losses(runs_data)
    plot_feature_stats(runs_data)

if __name__ == "__main__":
    main()
