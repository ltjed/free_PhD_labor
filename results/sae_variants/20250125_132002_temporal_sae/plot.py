import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Dictionary mapping run numbers to descriptive labels
labels = {
    "run_0": "Baseline",
    "run_1": "Hard Positional Masking",
    "run_2": "Soft Positional Masking", 
    "run_3": "Increased Sparsity",
    "run_4": "Position-Aware Loss",
    "run_5": "Position Groups",
    "run_6": "Attention Routing",
    "run_7": "Hierarchical Features",
    "run_8": "Contrastive Learning",
    "run_9": "Multi-Scale Integration",
    "run_10": "Residual Position Learning"
}

def load_run_results(run_dir):
    """Load results from a run directory."""
    info_path = os.path.join(run_dir, "final_info.json")
    if not os.path.exists(info_path):
        return None
    
    with open(info_path, 'r') as f:
        return json.load(f)

def plot_unlearning_scores():
    """Plot unlearning scores across runs."""
    scores = []
    run_labels = []
    
    for run_name in sorted(labels.keys()):
        results = load_run_results(run_name)
        if results and 'eval_result_metrics' in results:
            score = results['eval_result_metrics'].get('unlearning', {}).get('unlearning_score', None)
            if score is not None:
                scores.append(score)
                run_labels.append(labels[run_name])
    
    if not scores:
        print("No unlearning scores found")
        return
        
    plt.figure(figsize=(12, 6))
    sns.barplot(x=run_labels, y=scores)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Unlearning Score')
    plt.title('Unlearning Scores Across Experimental Runs')
    plt.tight_layout()
    plt.savefig('unlearning_scores.png')
    plt.close()

def plot_loss_curves():
    """Plot training loss curves for each run."""
    plt.figure(figsize=(12, 6))
    
    for run_name in sorted(labels.keys()):
        results_path = os.path.join(run_name, "all_results.npy")
        if not os.path.exists(results_path):
            continue
            
        results = np.load(results_path, allow_pickle=True).item()
        if 'training_log' in results:
            losses = [log.get('loss', None) for log in results['training_log'] if isinstance(log, dict)]
            if losses:
                plt.plot(losses, label=labels[run_name], alpha=0.7)
    
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Curves')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('loss_curves.png')
    plt.close()

def plot_final_metrics():
    """Plot final metrics comparison across runs."""
    metrics = {
        'final_loss': [],
        'labels': []
    }
    
    for run_name in sorted(labels.keys()):
        results = load_run_results(run_name)
        if results and 'final_loss' in results:
            metrics['final_loss'].append(results['final_loss'])
            metrics['labels'].append(labels[run_name])
    
    if not metrics['final_loss']:
        print("No final metrics found")
        return
        
    plt.figure(figsize=(12, 6))
    sns.barplot(x=metrics['labels'], y=metrics['final_loss'])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Final Loss')
    plt.title('Final Loss Comparison')
    plt.tight_layout()
    plt.savefig('final_metrics.png')
    plt.close()

if __name__ == "__main__":
    # Set style
    plt.style.use('default')
    sns.set_theme(style="whitegrid")
    sns.set_palette("husl", n_colors=len(labels))
    
    # Create plots
    plot_unlearning_scores()
    plot_loss_curves()
    plot_final_metrics()
    
    print("Plots generated:")
    print("- unlearning_scores.png")
    print("- loss_curves.png") 
    print("- final_metrics.png")
