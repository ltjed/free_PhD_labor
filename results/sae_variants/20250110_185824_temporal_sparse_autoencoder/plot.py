import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Style settings
plt.style.use('default')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#FF7F0E', '#2CA02C', '#1F77B4', '#9467BD', '#D62728'])

# Define labels for each run
labels = {
    'run_1': 'Core TSA (Full)',
    'run_2': 'TSA + Stability',
    'run_3': 'TSA Simplified',
    'run_4': 'Minimal Base',
    'run_5': 'Optimizer Focus'
}

def load_run_data(run_dir):
    """Load final_info.json data from a run directory."""
    try:
        with open(os.path.join(run_dir, 'final_info.json'), 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def plot_training_metrics():
    """Plot training steps and loss across runs."""
    steps = []
    losses = []
    run_names = []
    
    for run_dir, label in labels.items():
        data = load_run_data(run_dir)
        if data:
            steps.append(data['training_steps'])
            losses.append(data['final_loss'] if data['final_loss'] is not None else 0)
            run_names.append(label)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Training steps
    ax1.bar(run_names, steps)
    ax1.set_title('Training Steps Completed')
    ax1.set_xticklabels(run_names, rotation=45)
    ax1.set_ylabel('Steps')
    
    # Final loss
    ax2.bar(run_names, losses)
    ax2.set_title('Final Training Loss')
    ax2.set_xticklabels(run_names, rotation=45)
    ax2.set_ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

def plot_architecture_comparison():
    """Plot architecture-specific metrics across runs."""
    metrics = {
        'dict_size': [],
        'learning_rate': [],
        'sparsity_penalty': []
    }
    run_names = []
    
    for run_dir, label in labels.items():
        data = load_run_data(run_dir)
        if data:
            for metric in metrics:
                metrics[metric].append(data[metric])
            run_names.append(label)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (metric, values) in enumerate(metrics.items()):
        axes[i].bar(run_names, values)
        axes[i].set_title(metric.replace('_', ' ').title())
        axes[i].set_xticklabels(run_names, rotation=45)
    
    plt.tight_layout()
    plt.savefig('architecture_comparison.png')
    plt.close()

def main():
    """Generate all plots."""
    os.makedirs('plots', exist_ok=True)
    
    print("Generating training metrics plot...")
    plot_training_metrics()
    
    print("Generating architecture comparison plot...")
    plot_architecture_comparison()
    
    print("Plots saved in plots/ directory")

if __name__ == "__main__":
    main()
