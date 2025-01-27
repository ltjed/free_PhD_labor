import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path

# Define labels for each relevant run
labels = {
    "run_1": "Initial Top-k Implementation",
    "run_2": "Adaptive τ Implementation",
    "run_3": "Training Step Fix",
    "run_4": "Training Loop Debug",
    "run_5": "Data Pipeline Validation",
    "run_6": "Activation Buffer Debug",
    "run_7": "Activation Validation",
    "run_8": "Buffer Iterator Fix",
    "run_9": "Training Loop State Verification"
}

def load_run_results(run_dir):
    """Load results from a run directory."""
    results_path = os.path.join(run_dir, "all_results.npy")
    if not os.path.exists(results_path):
        return None
    return np.load(results_path, allow_pickle=True).item()

def plot_training_metrics():
    """Plot training metrics across all runs."""
    # Setup plots
    fig = plt.figure(figsize=(15, 20))
    gs = fig.add_gridspec(4, 1)
    
    # Tau History Plot
    ax1 = fig.add_subplot(gs[0])
    # Correlation Plot
    ax2 = fig.add_subplot(gs[1])
    # Training Steps Plot
    ax3 = fig.add_subplot(gs[2])
    # Loss Plot
    ax4 = fig.add_subplot(gs[3])
    
    # Plot data for each run
    for run_dir, label in labels.items():
        if not os.path.exists(run_dir):
            continue
            
        results = load_run_results(run_dir)
        if results is None:
            continue
            
        # Plot tau history
        if 'tau_history' in results:
            steps = range(len(results['tau_history']))
            ax1.plot(steps, results['tau_history'], label=f'{label}')
            
        # Plot correlation statistics
        if 'correlation_history' in results:
            steps = range(len(results['correlation_history']))
            means = [x[0] for x in results['correlation_history']]
            stds = [x[1] for x in results['correlation_history']]
            ax2.plot(steps, means, label=f'{label}')
            ax2.fill_between(steps, 
                           [m - s for m, s in zip(means, stds)],
                           [m + s for m, s in zip(means, stds)],
                           alpha=0.1)
                           
        # Plot training steps
        if 'training_log' in results:
            steps = range(len(results['training_log']))
            ax3.plot(steps, [i for i in range(len(steps))], label=f'{label}')
            
        # Plot loss
        if 'training_log' in results:
            losses = [log.get('loss', float('nan')) for log in results['training_log']]
            valid_losses = [l for l in losses if not np.isnan(l)]
            if valid_losses:
                ax4.plot(range(len(valid_losses)), valid_losses, label=f'{label}')
    
    # Customize plots
    ax1.set_title('Adaptive τ History')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('τ Value')
    ax1.grid(True)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax2.set_title('Feature Correlation Distribution')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Mean Correlation')
    ax2.grid(True)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax3.set_title('Training Progress')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Successful Steps')
    ax3.grid(True)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax4.set_title('Training Loss')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Loss')
    ax4.set_yscale('log')
    ax4.grid(True)
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_training_metrics()
