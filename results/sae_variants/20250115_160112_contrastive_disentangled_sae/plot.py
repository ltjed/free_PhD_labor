import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Define which runs to plot and their labels
labels = {
    "run_0": "Baseline",
    "run_1": "Initial Contrastive",
    "run_2": "Reduced Dict Size",
    "run_3": "Simplified Contrastive",
    "run_4": "Baseline SAE",
    "run_5": "Memory Optimized"
}

def load_run_data(run_dir):
    """Load results from a run directory"""
    final_info_path = os.path.join(run_dir, "final_info.json")
    if not os.path.exists(final_info_path):
        return None
        
    with open(final_info_path) as f:
        return json.load(f)

def plot_metrics(run_data):
    """Plot key metrics across runs"""
    # Extract metrics
    metrics = defaultdict(list)
    for run_id, label in labels.items():
        data = load_run_data(run_id)
        if data is None:
            continue
            
        metrics['run'].append(label)
        metrics['training_steps'].append(data.get('training_steps', 0))
        # Handle None values by converting to 0 for plotting
        final_loss = data.get('final_loss')
        metrics['final_loss'].append(0 if final_loss is None else final_loss)
        metrics['dict_size'].append(data.get('dict_size', 0))
        metrics['learning_rate'].append(data.get('learning_rate', 0))
        metrics['sparsity_penalty'].append(data.get('sparsity_penalty', 0))

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('SAE Training Experiment Results', fontsize=16)

    # Plot training progress
    axes[0,0].bar(metrics['run'], metrics['training_steps'], color='skyblue')
    axes[0,0].set_title('Training Steps Completed')
    axes[0,0].set_ylabel('Steps')
    axes[0,0].tick_params(axis='x', rotation=45)

    # Plot final loss
    axes[0,1].bar(metrics['run'], metrics['final_loss'], color='lightgreen')
    axes[0,1].set_title('Final Loss')
    axes[0,1].set_ylabel('Loss')
    axes[0,1].tick_params(axis='x', rotation=45)

    # Plot dictionary size
    axes[1,0].bar(metrics['run'], metrics['dict_size'], color='salmon')
    axes[1,0].set_title('Dictionary Size')
    axes[1,0].set_ylabel('Size')
    axes[1,0].tick_params(axis='x', rotation=45)

    # Plot hyperparameters
    axes[1,1].plot(metrics['run'], metrics['learning_rate'], marker='o', label='Learning Rate')
    axes[1,1].plot(metrics['run'], metrics['sparsity_penalty'], marker='o', label='Sparsity Penalty')
    axes[1,1].set_title('Hyperparameters')
    axes[1,1].set_ylabel('Value')
    axes[1,1].legend()
    axes[1,1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('experiment_results.png')
    plt.close()

def main():
    # Get list of all run directories
    run_dirs = [d for d in os.listdir() if d.startswith('run_') and os.path.isdir(d)]
    
    # Load data from each run
    all_data = {}
    for run_dir in run_dirs:
        if run_dir in labels:
            data = load_run_data(run_dir)
            if data:
                all_data[run_dir] = data

    # Generate plots
    plot_metrics(all_data)

if __name__ == "__main__":
    main()
