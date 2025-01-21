import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Dictionary mapping run directories to their display labels
labels = {
    'run_0': 'Baseline',
    'run_1': 'Initial MTSAE',
    'run_2': 'With Feature Separation',
    'run_3': 'Enhanced Debugging',
    'run_4': 'Gradient Diagnostics',
    'run_5': 'Final Run'
}

def load_run_data(run_dir):
    """Load training data from a run directory."""
    try:
        with open(os.path.join(run_dir, 'all_results.npy'), 'rb') as f:
            data = np.load(f, allow_pickle=True).item()
            print(data)
        return data.get('training_log', [])
    except:
        return []

def extract_metrics(log_data):
    """Extract relevant metrics from training log."""
    steps = []
    losses = []
    grad_norms = []
    conv_stats = []
    
    for entry in log_data:
        if isinstance(entry, dict):
            # Basic metrics
            if 'step' in entry.get('iteration_stats', {}):
                steps.append(entry['iteration_stats']['step'])
            if 'loss' in entry:
                losses.append(entry['loss'])
                
            # Gradient norms
            if 'gradient_norms' in entry:
                grad_norms.append(np.mean(list(entry['gradient_norms'].values())))
                
            # Convolution statistics
            if 'conv_stats' in entry:
                conv_means = np.mean([v for k, v in entry['conv_stats'].items() if 'mean' in k])
                conv_stats.append(conv_means)
    
    return {
        'steps': steps,
        'losses': losses,
        'grad_norms': grad_norms,
        'conv_stats': conv_stats
    }

def plot_training_curves():
    """Generate training curve plots for all runs."""
    plt.style.use('seaborn-v0_8-darkgrid')  # Using a specific seaborn-compatible style
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('MTSAE Training Analysis', fontsize=16)
    
    # Plot settings
    colors = sns.color_palette('husl', n_colors=len(labels))
    
    # Collect data from all runs
    all_metrics = {}
    for run_dir, label in labels.items():
        if os.path.exists(run_dir):
            log_data = load_run_data(run_dir)
            all_metrics[label] = extract_metrics(log_data)
    
    # Plot loss curves
    ax = axes[0, 0]
    has_data = False
    for (label, metrics), color in zip(all_metrics.items(), colors):
        if metrics['losses'] and len(metrics['losses']) > 0:
            ax.plot(metrics['steps'], metrics['losses'], 
                   label=label, color=color, alpha=0.8)
            has_data = True
    ax.set_title('Training Loss')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Loss')
    if has_data:
        ax.legend()
    
    # Plot gradient norms
    ax = axes[0, 1]
    has_data = False
    for (label, metrics), color in zip(all_metrics.items(), colors):
        if metrics['grad_norms'] and len(metrics['grad_norms']) > 0:
            ax.plot(metrics['steps'], metrics['grad_norms'],
                   label=label, color=color, alpha=0.8)
            has_data = True
    ax.set_title('Average Gradient Norms')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Norm')
    if has_data:
        ax.legend()
    
    # Plot convolution statistics
    ax = axes[1, 0]
    has_data = False
    for (label, metrics), color in zip(all_metrics.items(), colors):
        if metrics['conv_stats'] and len(metrics['conv_stats']) > 0:
            ax.plot(metrics['steps'], metrics['conv_stats'],
                   label=label, color=color, alpha=0.8)
            has_data = True
    ax.set_title('Temporal Convolution Statistics')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Mean Activation')
    if has_data:
        ax.legend()
    
    # Plot training duration comparison
    ax = axes[1, 1]
    durations = {label: len(metrics['steps']) for label, metrics in all_metrics.items()}
    bars = ax.bar(range(len(durations)), list(durations.values()), 
                 color=colors[:len(durations)])
    ax.set_title('Training Duration Comparison')
    ax.set_xlabel('Run')
    ax.set_ylabel('Steps Completed')
    ax.set_xticks(range(len(durations)))
    ax.set_xticklabels(list(durations.keys()), rotation=45)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('training_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_run_comparison():
    """Generate bar plots comparing final metrics across runs."""
    plt.style.use('seaborn-v0_8-darkgrid')  # Using a specific seaborn-compatible style
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Collect final metrics from each run
    final_metrics = {}
    for run_dir, label in labels.items():
        try:
            with open(os.path.join(run_dir, 'final_info.json'), 'r') as f:
                metrics = json.load(f)
                if any(metrics.get(key) is not None for key in ['training_steps', 'final_loss']):
                    final_metrics[label] = metrics
        except:
            continue
    
    if not final_metrics:
        print("No valid metrics found for comparison plot")
        plt.close(fig)
        return
        
    # Extract relevant metrics for comparison
    metric_names = ['training_steps', 'final_loss']
    metric_data = {metric: [] for metric in metric_names}
    run_labels = []
    
    for label, metrics in final_metrics.items():
        run_labels.append(label)
        for metric in metric_names:
            value = metrics.get(metric)
            metric_data[metric].append(0.0 if value is None else float(value))
    
    # Create grouped bar plot
    x = np.arange(len(run_labels))
    width = 0.35
    multiplier = 0
    
    for metric, values in metric_data.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, values, width, label=metric)
        ax.bar_label(rects, padding=3)
        multiplier += 1
    
    ax.set_ylabel('Value')
    ax.set_title('Final Metrics Comparison')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(run_labels, rotation=45)
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig('run_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

if __name__ == '__main__':
    plot_training_curves()
    plot_run_comparison()
