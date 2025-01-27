import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path
import seaborn as sns

# Set basic style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#99FFCC', '#FFB366', '#FF99FF']

# Dictionary mapping run numbers to their descriptive labels
labels = {
    "run_0": "Baseline (4-bit Quantization + JL)",
    "run_1": "Initial Curriculum",
    "run_2": "Dynamic Rank Curriculum", 
    "run_3": "Diversity-Based Thresholds",
    "run_4": "Feature-Weighted Loss",
    "run_5": "Adaptive Sparsity",
    "run_6": "Contrastive Learning",
    "run_7": "Multi-Scale Feature Pyramid"
}

def load_results(run_dir):
    """Load results from a run directory."""
    results_path = Path(run_dir) / "final_info.json"
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return None

def plot_accuracy_comparison():
    """Plot accuracy metrics across runs."""
    metrics = ['llm_test_accuracy', 'llm_top_1_test_accuracy', 'llm_top_5_test_accuracy']
    metric_labels = ['Overall Test Accuracy', 'Top-1 Accuracy', 'Top-5 Accuracy']
    
    runs = sorted([d for d in os.listdir('.') if d.startswith('run_') and d in labels])
    
    fig, ax = plt.subplots(figsize=(15, 8))
    x = np.arange(len(runs))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = []
        for run in runs:
            results = load_results(run)
            if results:
                values.append(results.get(metric, 0))
            else:
                values.append(0)
        
        ax.bar(x + i*width, values, width, label=metric_labels[i])
    
    ax.set_ylabel('Accuracy')
    ax.set_title('Performance Metrics Across Runs')
    ax.set_xticks(x + width)
    ax.set_xticklabels([labels[run] for run in runs], rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png')
    plt.close()

def plot_task_specific_performance():
    """Plot performance on specific tasks across runs."""
    tasks = ['Europarl', 'Amazon Reviews', 'GitHub Code', 'AG News']
    
    runs = sorted([d for d in os.listdir('.') if d.startswith('run_') and d in labels])
    
    fig, ax = plt.subplots(figsize=(15, 8))
    x = np.arange(len(runs))
    width = 0.2
    
    for i, task in enumerate(tasks):
        values = []
        for run in runs:
            results = load_results(run)
            if results and 'task_specific' in results:
                values.append(results['task_specific'].get(task, 0))
            else:
                values.append(0)
        
        ax.bar(x + i*width, values, width, label=task)
    
    ax.set_ylabel('Accuracy')
    ax.set_title('Task-Specific Performance Across Runs')
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels([labels[run] for run in runs], rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('task_specific_performance.png')
    plt.close()

def plot_efficiency_metrics():
    """Plot efficiency metrics across runs."""
    metrics = ['training_time', 'memory_usage', 'reconstruction_error']
    
    runs = sorted([d for d in os.listdir('.') if d.startswith('run_') and d in labels])
    
    fig, ax = plt.subplots(figsize=(15, 8))
    x = np.arange(len(runs))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = []
        for run in runs:
            results = load_results(run)
            if results and 'efficiency' in results:
                values.append(results['efficiency'].get(metric, 0))
            else:
                values.append(0)
        
        ax.bar(x + i*width, values, width, label=metric)
    
    ax.set_ylabel('Normalized Value')
    ax.set_title('Efficiency Metrics Across Runs')
    ax.set_xticks(x + width)
    ax.set_xticklabels([labels[run] for run in runs], rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('efficiency_metrics.png')
    plt.close()

def main():
    """Generate all plots."""
    print("Generating accuracy comparison plot...")
    plot_accuracy_comparison()
    
    print("Generating task-specific performance plot...")
    plot_task_specific_performance()
    
    print("Generating efficiency metrics plot...")
    plot_efficiency_metrics()
    
    print("All plots generated successfully!")

if __name__ == "__main__":
    main()
