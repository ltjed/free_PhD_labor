import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Dictionary mapping run directories to their display labels
labels = {
    'run_9': 'Direct Feature Selection',
    'run_10': 'Baseline Training'
}

def load_run_data(run_dir):
    """Load training data from a run directory."""
    info_path = os.path.join(run_dir, 'final_info.json')
    results_path = os.path.join(run_dir, 'all_results.npy')
    
    data = {}
    
    # Load final info if available
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            data['final_info'] = json.load(f)
    
    # Load training results if available        
    if os.path.exists(results_path):
        with open(results_path, 'rb') as f:
            try:
                results = np.load(f, allow_pickle=True).item()
                data['training_log'] = results.get('training_log', [])
                data['config'] = results.get('config', {})
            except Exception as e:
                print(f"Error loading results from {run_dir}: {e}")
                data['training_log'] = []
                data['config'] = {}
    
    return data

def plot_memory_usage(runs_data):
    """Plot GPU memory usage over training steps."""
    plt.figure(figsize=(10, 6))
    has_data = False
    
    for run_name, data in runs_data.items():
        if run_name not in labels:
            continue
            
        training_log = data.get('training_log', [])
        if not training_log:
            continue
            
        steps = []
        allocated = []
        cached = []
        
        for step, log in enumerate(training_log):
            if isinstance(log, dict) and isinstance(log.get('losses'), dict):
                # Memory stats might be in the losses dict
                mem_stats = log.get('losses', {}).get('memory_stats', {})
                if mem_stats:
                    steps.append(step)
                    allocated.append(mem_stats.get('allocated', 0))
                    cached.append(mem_stats.get('cached', 0))
        
        if steps:
            has_data = True
            plt.plot(steps, allocated, label=f"{labels[run_name]} (Allocated)", linestyle='-')
            plt.plot(steps, cached, label=f"{labels[run_name]} (Cached)", linestyle='--')
    
    plt.xlabel('Training Step')
    plt.ylabel('Memory Usage (MB)')
    plt.title('GPU Memory Usage During Training')
    if has_data:
        plt.legend()
    plt.grid(True)
    plt.savefig('memory_usage.png')
    plt.close()

def plot_training_metrics(runs_data):
    """Plot training metrics (loss, sparsity) over steps."""
    metrics = ['l2_loss', 'sparsity_loss', 'loss']
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 12), sharex=True)
    
    for metric, ax in zip(metrics, axes):
        for run_name, data in runs_data.items():
            if run_name not in labels:
                continue
                
            training_log = data.get('training_log', [])
            if not training_log:
                continue
                
            steps = []
            values = []
            
            for step, log in enumerate(training_log):
                if isinstance(log, dict) and 'losses' in log and metric in log['losses']:
                    steps.append(step)
                    values.append(log['losses'][metric])
            
            if steps:
                ax.plot(steps, values, label=labels[run_name])
        
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.grid(True)
        ax.legend()
    
    axes[-1].set_xlabel('Training Step')
    plt.suptitle('Training Metrics Over Time')
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

def plot_final_stats(runs_data):
    """Plot final statistics comparison between runs."""
    metrics = ['training_steps', 'final_loss', 'learning_rate', 'sparsity_penalty']
    runs = [run for run in runs_data.keys() if run in labels]
    
    if not runs:
        return
        
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.8 / len(runs)
    
    has_data = False
    for i, run in enumerate(runs):
        values = []
        valid_values = False
        for metric in metrics:
            value = runs_data[run].get('final_info', {}).get(metric)
            if value is not None:
                values.append(float(value))
                valid_values = True
            else:
                values.append(0.0)
        
        if valid_values:
            has_data = True
            offset = width * (i - len(runs)/2 + 0.5)
            ax.bar(x + offset, values, width, label=labels[run])
    
    ax.set_ylabel('Value')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45)
    ax.legend()
    plt.title('Final Statistics Comparison')
    plt.tight_layout()
    plt.savefig('final_stats.png')
    plt.close()

def main():
    # Set style
    plt.style.use('default')
    plt.rcParams['axes.grid'] = True
    plt.rcParams['lines.linewidth'] = 2
    
    # Load data from all runs
    runs_data = {}
    for run_dir in os.listdir('.'):
        if run_dir.startswith('run_') and os.path.isdir(run_dir):
            runs_data[run_dir] = load_run_data(run_dir)
    
    # Generate plots
    plot_memory_usage(runs_data)
    plot_training_metrics(runs_data)
    plot_final_stats(runs_data)

if __name__ == '__main__':
    main()
