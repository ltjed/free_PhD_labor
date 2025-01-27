import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path

# Dictionary mapping run directories to their display labels
labels = {
    'run_0': 'Baseline SAE',
    'run_9': 'Aggressive Competition',
    'run_10': 'Conservative Competition'
}

def load_results(run_dir):
    """Load results from a run directory."""
    info_path = os.path.join(run_dir, 'final_info.json')
    if not os.path.exists(info_path):
        return None
    with open(info_path, 'r') as f:
        return json.load(f)

def plot_metrics_comparison():
    """Plot comparison of key metrics across runs."""
    metrics = {
        'reconstruction': ['explained_variance', 'mse', 'cossim'],
        'sparsity': ['l0', 'l1'],
        'model_preservation': ['kl_div_score', 'ce_loss_score']
    }
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    fig.suptitle('Comparison of Key Metrics Across Runs', fontsize=16)
    
    for idx, (metric_group, metric_list) in enumerate(metrics.items()):
        x_pos = np.arange(len(metric_list))
        width = 0.8 / len(labels)
        
        for i, (run_dir, label) in enumerate(labels.items()):
            results = load_results(run_dir)
            if results is None:
                continue
                
            values = []
            for metric in metric_list:
                try:
                    value = results.get('core evaluation results', {}).get('metrics', {}).get(metric_group, {}).get(metric, 0)
                    values.append(value)
                except (KeyError, AttributeError):
                    values.append(0)
            
            axes[idx].bar(x_pos + i*width, values, width, label=label)
        
        axes[idx].set_ylabel('Value')
        axes[idx].set_title(f'{metric_group.replace("_", " ").title()} Metrics')
        axes[idx].set_xticks(x_pos + width/2)
        axes[idx].set_xticklabels([m.replace('_', ' ').title() for m in metric_list])
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('metrics_comparison.png')
    plt.close()

def plot_scr_metrics():
    """Plot Selective Concept Response (SCR) metrics."""
    thresholds = [2, 10, 50]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(thresholds))
    width = 0.8 / len(labels)
    
    for i, (run_dir, label) in enumerate(labels.items()):
        results = load_results(run_dir)
        if results is None:
            continue
            
        values = []
        for threshold in thresholds:
            try:
                value = results.get('scr and tpp evaluations results', {}).get('eval_result_metrics', {}).get('scr_metrics', {}).get(f'threshold_{threshold}', 0)
                values.append(value)
            except (KeyError, AttributeError):
                values.append(0)
        
        ax.bar(x_pos + i*width, values, width, label=label)
    
    ax.set_ylabel('SCR Score')
    ax.set_title('Selective Concept Response Metrics')
    ax.set_xticks(x_pos + width/2)
    ax.set_xticklabels([f'Threshold {t}' for t in thresholds])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('scr_metrics.png')
    plt.close()

def plot_temperature_scaling():
    """Plot the adaptive temperature scaling function with annealing."""
    density_range = np.linspace(0, 1, 100)
    steps = [0, 500, 1000, 2000]
    base_temp = 0.1
    min_temp = 0.05
    max_temp = 0.3
    
    plt.figure(figsize=(10, 6))
    
    for step in steps:
        annealing_factor = 0.5 * (1 + np.tanh(step/2000 - 1))
        density_factor = 1.0 + (np.exp(np.exp(density_range) - 1) - 1) * 0.3
        temperatures = base_temp * annealing_factor * density_factor
        temperatures = np.clip(temperatures, min_temp, max_temp)
        
        plt.plot(density_range, temperatures, label=f'Step {step}')
    
    plt.xlabel('Activation Density')
    plt.ylabel('Competition Temperature')
    plt.title('Adaptive Temperature Scaling with Annealing')
    plt.legend()
    plt.grid(True)
    plt.savefig('temperature_scaling.png')
    plt.close()

if __name__ == "__main__":
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Generate all plots
    plot_metrics_comparison()
    plot_scr_metrics()
    plot_temperature_scaling()
