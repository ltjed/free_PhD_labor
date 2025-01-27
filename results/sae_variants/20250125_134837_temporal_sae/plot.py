import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path
import seaborn as sns

# Dictionary mapping run numbers to their descriptive labels
labels = {
    "run_0": "Baseline",
    "run_1": "Hard Positional Masking",
    "run_2": "Soft Positional Masking",
    "run_3": "Optimized Temperature",
    "run_4": "Hybrid Content-Position",
    "run_5": "Adaptive Mask Evolution",
    "run_6": "Attention-Based Routing",
    "run_7": "Contrastive Learning",
    "run_8": "Hierarchical Features",
    "run_9": "Dynamic Allocation",
    "run_10": "Information-Theoretic"
}

def load_run_data(run_dir):
    """Load results from a run directory."""
    try:
        with open(os.path.join(run_dir, "final_info.json"), "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def plot_unlearning_scores():
    """Plot unlearning scores across runs."""
    scores = []
    run_labels = []
    
    for run_name, label in labels.items():
        data = load_run_data(run_name)
        if data and "eval_result_metrics" in data:
            if "unlearning" in data["eval_result_metrics"]:
                score = data["eval_result_metrics"]["unlearning"]["unlearning_score"]
                scores.append(score)
                run_labels.append(label)
    
    if scores:
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(scores)), scores)
        plt.xticks(range(len(scores)), run_labels, rotation=45, ha='right')
        plt.ylabel('Unlearning Score')
        plt.title('Unlearning Performance Across Runs')
        plt.tight_layout()
        plt.savefig('unlearning_scores.png')
        plt.close()

def plot_loss_curves():
    """Plot training loss curves for each run."""
    plt.figure(figsize=(12, 6))
    
    for run_name, label in labels.items():
        try:
            results = np.load(os.path.join(run_name, "all_results.npy"), allow_pickle=True).item()
            if "training_log" in results:
                losses = [log.get("loss", None) for log in results["training_log"] if isinstance(log, dict)]
                if losses:
                    plt.plot(losses, label=label)
        except (FileNotFoundError, ValueError):
            continue
    
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Curves')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('loss_curves.png')
    plt.close()

def plot_feature_usage():
    """Plot feature usage distribution for the final state of each run."""
    plt.figure(figsize=(12, 6))
    
    for i, (run_name, label) in enumerate(labels.items()):
        try:
            results = np.load(os.path.join(run_name, "all_results.npy"), allow_pickle=True).item()
            if "final_info" in results and "feature_usage" in results["final_info"]:
                usage = results["final_info"]["feature_usage"]
                plt.subplot(2, 5, i+1)
                sns.histplot(usage, bins=30)
                plt.title(label)
        except (FileNotFoundError, ValueError):
            continue
    
    plt.tight_layout()
    plt.savefig('feature_usage.png')
    plt.close()

def main():
    """Generate all plots."""
    # Set style
    plt.style.use('seaborn')
    
    # Create plots
    plot_unlearning_scores()
    plot_loss_curves()
    plot_feature_usage()

if __name__ == "__main__":
    main()
