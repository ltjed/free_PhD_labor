import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

# Style settings
plt.style.use('default')
plt.rcParams.update({
    'figure.figsize': [12, 8],
    'figure.dpi': 100,
    'axes.prop_cycle': plt.cycler('color', ['#FF0000', '#0000FF', '#008000', '#FF00FF', '#FFA500', '#800080']),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.facecolor': '#F0F0F0'
})

# Dictionary mapping run directories to their display labels
labels = {
    'run_1': 'Initial Training Attempt',
    'run_2': 'Increased Training Tokens',
    'run_3': 'Loss Tracking Implementation',
    'run_4': 'Training Loop Completion',
    'run_5': 'Loss Value Propagation'
}

def load_training_data(run_dir):
    """Load training data from a run directory."""
    try:
        results_path = Path(run_dir) / "all_results.npy"
        if results_path.exists():
            results = np.load(results_path, allow_pickle=True).item()
            return results.get('training_log', [])
    except Exception as e:
        print(f"Error loading data from {run_dir}: {e}")
        return []

def extract_loss_values(training_log):
    """Extract loss values from training log."""
    steps = []
    l2_losses = []
    l1_losses = []
    total_losses = []
    
    for step, entry in enumerate(training_log):
        if isinstance(entry, dict):
            steps.append(step)
            l2_losses.append(entry.get('l2_loss', None))
            l1_losses.append(entry.get('l1_loss', None))
            total_losses.append(entry.get('total_loss', None))
    
    return steps, l2_losses, l1_losses, total_losses

def plot_training_curves():
    """Generate training curves for all runs."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    for run_dir, label in labels.items():
        training_log = load_training_data(run_dir)
        if not training_log:
            continue
            
        steps, l2_losses, l1_losses, total_losses = extract_loss_values(training_log)
        
        # Plot L2 reconstruction loss
        if any(x is not None for x in l2_losses):
            ax1.plot(steps, l2_losses, label=f'{label} - L2 Loss')
        
        # Plot L1 sparsity loss
        if any(x is not None for x in l1_losses):
            ax2.plot(steps, l1_losses, label=f'{label} - L1 Loss')
        
        # Plot total loss
        if any(x is not None for x in total_losses):
            ax3.plot(steps, total_losses, label=f'{label} - Total Loss')
    
    ax1.set_title('L2 Reconstruction Loss Over Training Steps')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('L2 Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_title('L1 Sparsity Loss Over Training Steps')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('L1 Loss')
    ax2.legend()
    ax2.grid(True)
    
    ax3.set_title('Total Loss Over Training Steps')
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Total Loss')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

def plot_final_metrics():
    """Generate bar plots for final metrics across runs."""
    final_metrics = {}
    
    for run_dir, label in labels.items():
        try:
            with open(Path(run_dir) / "final_info.json") as f:
                info = json.load(f)
                final_metrics[label] = info
        except Exception as e:
            print(f"Error loading final metrics from {run_dir}: {e}")
            continue
    
    if not final_metrics:
        return
    
    # Plot training steps comparison
    plt.figure(figsize=(10, 6))
    steps = [metrics.get('training_steps', 0) for metrics in final_metrics.values()]
    plt.bar(final_metrics.keys(), steps)
    plt.title('Training Steps Comparison')
    plt.xticks(rotation=45)
    plt.ylabel('Number of Steps')
    plt.tight_layout()
    plt.savefig('training_steps_comparison.png')
    plt.close()
    
    # Plot final loss comparison (where available)
    plt.figure(figsize=(10, 6))
    losses = [metrics.get('final_loss', 0) for metrics in final_metrics.values()]
    if any(loss is not None for loss in losses):
        plt.bar(final_metrics.keys(), losses)
        plt.title('Final Loss Comparison')
        plt.xticks(rotation=45)
        plt.ylabel('Final Loss')
        plt.tight_layout()
        plt.savefig('final_loss_comparison.png')
    plt.close()

def main():
    """Generate all plots."""
    print("Generating training curves...")
    plot_training_curves()
    
    print("Generating final metrics comparison...")
    plot_final_metrics()
    
    print("Plots have been saved to:")
    print("- training_curves.png")
    print("- training_steps_comparison.png")
    print("- final_loss_comparison.png")

if __name__ == "__main__":
    main()
