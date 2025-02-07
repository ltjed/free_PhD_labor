import matplotlib.pyplot as plt
import numpy as np
import json
import os
from collections import defaultdict

# Dictionary mapping run names to their display labels
labels = {
    "run_0": "Baseline SAE",
    "run_1": "Initial Temporal",
    "run_2": "Improved Stability", 
    "run_3": "Dynamic Resampling",
    "run_4": "Feature-wise Scaling"
}

def load_results(run_dir):
    """Load results from a run directory."""
    results_path = os.path.join(run_dir, "final_info.json")
    if not os.path.exists(results_path):
        return None
    with open(results_path, 'r') as f:
        return json.load(f)

def plot_absorption_scores(results_by_run):
    """Plot absorption scores across runs."""
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(labels))
    scores = []
    for run in labels.keys():
        if run in results_by_run:
            result = results_by_run[run]
            if 'absorption evaluation results' in result:
                score = result['absorption evaluation results']['eval_result_metrics']['mean']['mean_absorption_score']
                scores.append(score)
            else:
                scores.append(0)
    
    plt.bar(x, scores)
    plt.xticks(x, labels.values(), rotation=45)
    plt.ylabel('Mean Absorption Score')
    plt.title('Absorption Scores Across Different Runs')
    plt.tight_layout()
    plt.savefig('absorption_scores.png')
    plt.close()

def plot_reconstruction_quality(results_by_run):
    """Plot reconstruction quality metrics."""
    plt.figure(figsize=(12, 6))
    
    metrics = ['explained_variance', 'mse', 'cossim']
    x = np.arange(len(labels))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = []
        for run in labels.keys():
            if run in results_by_run:
                result = results_by_run[run]
                if 'core evaluation results' in result:
                    value = result['core evaluation results']['metrics']['reconstruction_quality'][metric]
                    values.append(value)
                else:
                    values.append(0)
        plt.bar(x + i*width, values, width, label=metric)
    
    plt.xlabel('Runs')
    plt.ylabel('Score')
    plt.title('Reconstruction Quality Metrics')
    plt.xticks(x + width, labels.values(), rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('reconstruction_quality.png')
    plt.close()

def plot_training_loss(results_by_run):
    """Plot final training loss across runs."""
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(labels))
    losses = []
    for run in labels.keys():
        if run in results_by_run:
            result = results_by_run[run]
            if 'final_info' in result:
                loss = result['final_info']['final_loss']
                losses.append(loss)
            else:
                losses.append(0)
    
    plt.bar(x, losses)
    plt.xticks(x, labels.values(), rotation=45)
    plt.ylabel('Final Training Loss')
    plt.title('Training Loss Comparison')
    plt.tight_layout()
    plt.savefig('training_loss.png')
    plt.close()

def plot_sparsity_metrics(results_by_run):
    """Plot sparsity metrics (L0 and L1 norms)."""
    plt.figure(figsize=(12, 6))
    
    metrics = ['l0', 'l1']
    x = np.arange(len(labels))
    width = 0.35
    
    for i, metric in enumerate(metrics):
        values = []
        for run in labels.keys():
            if run in results_by_run:
                result = results_by_run[run]
                if 'core evaluation results' in result:
                    value = result['core evaluation results']['metrics']['sparsity'][metric]
                    values.append(value)
                else:
                    values.append(0)
        plt.bar(x + i*width, values, width, label=f'{metric} norm')
    
    plt.xlabel('Runs')
    plt.ylabel('Value')
    plt.title('Sparsity Metrics')
    plt.xticks(x + width/2, labels.values(), rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('sparsity_metrics.png')
    plt.close()

def main():
    # Load results from each run
    results_by_run = {}
    for run_name in labels.keys():
        results = load_results(run_name)
        if results:
            results_by_run[run_name] = results
    
    # Generate plots
    plot_absorption_scores(results_by_run)
    plot_reconstruction_quality(results_by_run)
    plot_training_loss(results_by_run)
    plot_sparsity_metrics(results_by_run)

if __name__ == "__main__":
    main()
