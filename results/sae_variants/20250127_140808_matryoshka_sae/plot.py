import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path

# Style settings
plt.style.use('default')
colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

# Define labels for each run
labels = {
    "run_0": "Baseline (Standard SAE)",
    "run_3": "α=0.5 (Equal Split)",
    "run_4": "α=0.3 (Reduced Shared)",
    "run_5": "α=0.1 (Minimal Shared)", 
    "run_6": "α=0.2 (Optimal)"
}

def load_results(run_dir):
    """Load results from a run directory."""
    try:
        with open(os.path.join(run_dir, "final_info.json"), 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"No results found in {run_dir}")
        return None

def plot_metrics_comparison():
    """Plot comparison of key metrics across runs."""
    metrics = {
        'KL Divergence': [],
        'Explained Variance': [],
        'Cosine Similarity': [],
        'L0 Sparsity': [],
        'Training Loss': []
    }
    
    runs = []
    for run_name in labels.keys():
        results = load_results(run_name)
        if results:
            # Extract metrics from the results
            layer_key = list(results.keys())[0]  # Get first layer results
            run_results = results[layer_key]
            
            metrics['KL Divergence'].append(float(run_results['final_info'].get('kl_divergence', 0)))
            metrics['Explained Variance'].append(float(run_results['final_info'].get('explained_variance', 0)))
            metrics['Cosine Similarity'].append(float(run_results['final_info'].get('cosine_similarity', 0)))
            metrics['L0 Sparsity'].append(float(run_results['final_info'].get('l0_sparsity', 0)))
            metrics['Training Loss'].append(float(run_results['final_info'].get('final_loss', 0)))
            runs.append(labels[run_name])

    # Create subplots for each metric
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (metric_name, values) in enumerate(metrics.items()):
        if idx < len(axes):
            ax = axes[idx]
            ax.bar(runs, values)
            ax.set_title(metric_name)
            ax.tick_params(axis='x', rotation=45)
            ax.set_ylabel('Value')
    
    plt.tight_layout()
    plt.savefig('metrics_comparison.png')
    plt.close()

def plot_scr_metrics():
    """Plot SCR metrics across different thresholds."""
    thresholds = [2, 5, 10, 20, 50, 100]
    scr_values = {run_name: [] for run_name in labels.keys()}
    
    for run_name in labels.keys():
        results = load_results(run_name)
        if results:
            layer_key = list(results.keys())[0]
            run_results = results[layer_key]
            
            # Extract SCR values for each threshold
            for threshold in thresholds:
                scr_key = f'scr_n{threshold}'
                scr_values[run_name].append(
                    float(run_results['final_info'].get(scr_key, 0))
                )
    
    plt.figure(figsize=(12, 6))
    for run_name, values in scr_values.items():
        if values:  # Only plot if we have values
            plt.plot(thresholds, values, marker='o', label=labels[run_name])
    
    plt.xlabel('Threshold (n)')
    plt.ylabel('SCR Score')
    plt.title('SCR Performance Across Thresholds')
    plt.legend()
    plt.grid(True)
    plt.savefig('scr_metrics.png')
    plt.close()

def plot_training_progression():
    """Plot training loss progression."""
    plt.figure(figsize=(12, 6))
    
    for run_name in labels.keys():
        results = load_results(run_name)
        if results:
            layer_key = list(results.keys())[0]
            run_results = results[layer_key]
            
            # Extract training progression if available
            if 'training_log' in run_results:
                steps = []
                losses = []
                for step, metrics in run_results['training_log'].items():
                    if isinstance(metrics, dict) and 'loss' in metrics:
                        step_num = int(step.split()[-1])
                        steps.append(step_num)
                        losses.append(float(metrics['loss']))
                
                if steps and losses:
                    plt.plot(steps, losses, label=labels[run_name])
    
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Progression')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_progression.png')
    plt.close()

def main():
    """Generate all plots."""
    print("Generating plots...")
    plot_metrics_comparison()
    plot_scr_metrics()
    plot_training_progression()
    print("Plots generated successfully!")

if __name__ == "__main__":
    main()
