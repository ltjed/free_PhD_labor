import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Dictionary mapping run directories to their display labels
labels = {
    "run_1": "Self-Aligning SAE",
    "run_2": "Enhanced Adversarial",
    "run_3": "Increased Disentanglement", 
    "run_4": "Hierarchical SAE",
    "run_5": "Temporal-Aware SAE",
    "run_6": "Temporal Contrastive",
    "run_7": "Attention-Based Routing",
    "run_8": "Residual Gating",
    "run_9": "Adaptive Feature Pruning"
}

def load_results(run_dir):
    """Load results from a run directory."""
    info_path = os.path.join(run_dir, "final_info.json")
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            return json.load(f)
    return None

def plot_unlearning_scores():
    """Plot unlearning scores across runs."""
    scores = []
    run_names = []
    
    for run_dir, label in labels.items():
        results = load_results(run_dir)
        if results and 'eval_result_metrics' in results:
            if 'unlearning' in results['eval_result_metrics']:
                score = results['eval_result_metrics']['unlearning']['unlearning_score']
                scores.append(score)
                run_names.append(label)
    
    plt.figure(figsize=(12, 6))
    plt.bar(run_names, scores)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Unlearning Score')
    plt.title('Unlearning Performance Across Different SAE Variants')
    plt.tight_layout()
    plt.savefig('unlearning_scores.png')
    plt.close()

def plot_training_curves():
    """Plot training loss curves for each run."""
    plt.figure(figsize=(12, 6))
    
    for run_dir, label in labels.items():
        results_path = os.path.join(run_dir, "all_results.npy")
        if os.path.exists(results_path):
            with open(results_path, 'rb') as f:
                results = np.load(f, allow_pickle=True).item()
                if 'training_log' in results:
                    losses = [log['loss'] for log in results['training_log'] if 'loss' in log]
                    plt.plot(losses, label=label)
    
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Curves')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

def plot_feature_statistics():
    """Plot feature activation statistics."""
    plt.figure(figsize=(12, 6))
    
    for run_dir, label in labels.items():
        results_path = os.path.join(run_dir, "all_results.npy")
        if os.path.exists(results_path):
            with open(results_path, 'rb') as f:
                results = np.load(f, allow_pickle=True).item()
                if 'final_info' in results:
                    info = results['final_info']
                    if 'dict_size' in info:
                        plt.scatter([label], [info['dict_size']], 
                                 label=f"Dict Size: {info['dict_size']}")
    
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Dictionary Size')
    plt.title('Feature Dictionary Size Comparison')
    plt.tight_layout()
    plt.savefig('feature_statistics.png')
    plt.close()

def main():
    """Generate all plots."""
    os.makedirs('plots', exist_ok=True)
    
    # Set style
    plt.style.use('default')
    
    # Generate plots
    plot_unlearning_scores()
    plot_training_curves()
    plot_feature_statistics()
    
    print("Plots generated successfully in plots/ directory")

if __name__ == "__main__":
    main()
