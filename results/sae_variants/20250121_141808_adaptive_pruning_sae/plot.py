import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Configure plot style
plt.style.use('default')
sns.set_theme(style="whitegrid")
sns.set_palette("husl")
plt.rcParams.update({
    'figure.figsize': [12, 8],
    'font.size': 12,
    'axes.grid': True,
    'grid.alpha': 0.3
})

# Dictionary mapping run directories to their display labels
labels = {
    'run_0': 'Baseline SAE',
    'run_1': 'Dual-Dataset Tracking',
    'run_2': 'Gradient-Guided Features', 
    'run_3': 'Enhanced Feature Separation',
    'run_4': 'Hierarchical Organization',
    'run_5': 'Per-Feature Attention'
}

def load_results(run_dir):
    """Load results from a run directory."""
    results_path = Path(run_dir) / "final_info.json"
    if not results_path.exists():
        return None
    
    with open(results_path) as f:
        return json.load(f)

def plot_unlearning_scores():
    """Plot unlearning scores across different runs."""
    scores = []
    run_names = []
    
    for run_dir, label in labels.items():
        results = load_results(run_dir)
        if results and 'eval_result_metrics' in results:
            metrics = results['eval_result_metrics']
            if isinstance(metrics, dict) and 'unlearning' in metrics.get('eval_result_metrics', {}):
                score = metrics['eval_result_metrics']['unlearning']['unlearning_score']
                scores.append(score)
                run_names.append(label)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(run_names, scores)
    plt.title('Unlearning Performance Across Different Approaches')
    plt.ylabel('Unlearning Score')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('unlearning_scores.png')
    plt.close()

def plot_loss_curves():
    """Plot training loss curves for different runs."""
    plt.figure(figsize=(12, 6))
    
    for run_dir, label in labels.items():
        results_path = Path(run_dir) / "all_results.npy"
        if not results_path.exists():
            continue
            
        results = np.load(results_path, allow_pickle=True).item()
        if 'training_log' in results:
            losses = [log['loss'] for log in results['training_log'] if 'loss' in log]
            plt.plot(losses, label=label)
    
    plt.title('Training Loss Curves')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_losses.png')
    plt.close()

def plot_feature_importance():
    """Plot feature importance distributions for the latest run."""
    latest_run = max(labels.keys())
    results_path = Path(latest_run) / "all_results.npy"
    
    if not results_path.exists():
        return
        
    results = np.load(results_path, allow_pickle=True).item()
    if 'final_info' not in results:
        return
        
    retain_importance = results['final_info'].get('retain_importance', [])
    unlearn_importance = results['final_info'].get('unlearn_importance', [])
    
    if len(retain_importance) > 0 and len(unlearn_importance) > 0:
        plt.figure(figsize=(12, 6))
        plt.hist(retain_importance, alpha=0.5, label='Retain Features', bins=50)
        plt.hist(unlearn_importance, alpha=0.5, label='Unlearn Features', bins=50)
        plt.title('Feature Importance Distribution')
        plt.xlabel('Importance Score')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()

def main():
    """Generate all plots."""
    plot_unlearning_scores()
    plot_loss_curves()
    plot_feature_importance()

if __name__ == "__main__":
    main()
