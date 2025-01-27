import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Configure plot style
plt.style.use('default')
plt.rcParams['axes.grid'] = True
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])

# Define labels for each experimental run
labels = {
    "run_0": "Baseline",
    "run_1": "Hard Positional Masking",
    "run_2": "Soft Positional Masking", 
    "run_3": "Position-Specific Learning Rates",
    "run_4": "Combined Approach"
}

def load_results(run_dir):
    """Load results from a run directory."""
    results_path = os.path.join(run_dir, "final_info.json")
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return json.load(f)
    return None

def plot_top_k_accuracy(results_by_run):
    """Plot top-k accuracy comparison across runs."""
    plt.figure(figsize=(12, 6))
    
    k_values = [1, 2, 5, 10, 20, 50]
    
    for run_name, results in results_by_run.items():
        if run_name in labels:
            accuracies = []
            for k in k_values:
                key = f'llm_top_{k}_test_accuracy'
                if key in results:
                    accuracies.append(results[key])
            if accuracies:
                plt.plot(k_values, accuracies, marker='o', label=labels[run_name])
    
    plt.xlabel('k')
    plt.ylabel('Top-k Accuracy')
    plt.title('Top-k Accuracy Comparison Across Runs')
    plt.legend()
    plt.grid(True)
    plt.savefig('top_k_accuracy.png')
    plt.close()

def plot_task_specific_accuracy(results_by_run):
    """Plot task-specific accuracy comparison."""
    plt.figure(figsize=(15, 6))
    
    # Extract task names and accuracies
    tasks = ['europarl', 'amazon_reviews', 'ag_news', 'github-code']
    task_accuracies = defaultdict(list)
    
    for run_name, results in results_by_run.items():
        if run_name in labels:
            for task in tasks:
                key = f'llm_top_1_test_accuracy_{task}'
                if key in results:
                    task_accuracies[task].append((labels[run_name], results[key]))
    
    # Plot grouped bar chart
    x = np.arange(len(tasks))
    width = 0.15
    
    for i, (run_name, _) in enumerate(labels.items()):
        accuracies = []
        for task in tasks:
            matching = [acc for name, acc in task_accuracies[task] if name == labels[run_name]]
            accuracies.append(matching[0] if matching else 0)
        
        plt.bar(x + i*width, accuracies, width, label=labels[run_name])
    
    plt.xlabel('Tasks')
    plt.ylabel('Top-1 Accuracy')
    plt.title('Task-Specific Performance Comparison')
    plt.xticks(x + width*2, tasks, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('task_specific_accuracy.png')
    plt.close()

def main():
    # Load results from all runs
    results_by_run = {}
    for run_name in labels.keys():
        results = load_results(run_name)
        if results:
            results_by_run[run_name] = results
    
    # Generate plots
    plot_top_k_accuracy(results_by_run)
    plot_task_specific_accuracy(results_by_run)

if __name__ == "__main__":
    main()
