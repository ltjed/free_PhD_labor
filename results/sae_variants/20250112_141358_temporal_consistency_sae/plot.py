import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Dictionary mapping run numbers to their labels/descriptions
labels = {
    "run_1": "Initial Temporal Consistency",
    "run_2": "Balanced Loss Implementation", 
    "run_3": "Architectural Redesign",
    "run_4": "Training Loop Refinement",
    "run_5": "Memory-Optimized Training"
}

def load_run_data(run_dir):
    """Load training data and final info for a given run directory."""
    try:
        # Load results file
        results_path = os.path.join(run_dir, "all_results.npy")
        if os.path.exists(results_path):
            with open(results_path, 'rb') as f:
                data = np.load(f, allow_pickle=True).item()
            training_log = data.get('training_log', [])
        else:
            training_log = []
            
        # Load final info
        info_path = os.path.join(run_dir, "final_info.json")
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                final_info = json.load(f)
        else:
            final_info = {}
            
        return training_log, final_info
    except Exception as e:
        print(f"Error loading data from {run_dir}: {str(e)}")
        return [], {}

def plot_training_curves():
    """Plot training loss curves for all runs."""
    plt.figure(figsize=(12, 6))
    
    has_data = False
    for run_name, label in labels.items():
        if not os.path.exists(run_name):
            continue
            
        training_log, _ = load_run_data(run_name)
        
        if training_log:
            steps = [log['step'] for log in training_log]
            losses = [log['loss'] for log in training_log if log['loss'] is not None]
            if steps and losses:
                plt.plot(steps[:len(losses)], losses, label=label, alpha=0.8)
                has_data = True
    
    if not has_data:
        plt.text(0.5, 0.5, 'No training data available', 
                horizontalalignment='center', verticalalignment='center')
    
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Across Runs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_final_metrics():
    """Plot final metrics comparison across runs."""
    metrics = []
    run_labels = []
    
    for run_name, label in labels.items():
        if not os.path.exists(run_name):
            continue
            
        _, final_info = load_run_data(run_name)
        
        if final_info:
            training_steps = final_info.get('training_steps', 0)
            final_loss = final_info.get('final_loss')
            
            # Only append if we have valid data
            if training_steps is not None and final_loss is not None:
                metrics.append({
                    'training_steps': training_steps,
                    'final_loss': float(final_loss),
                })
                run_labels.append(label)
    
    if not metrics:
        plt.figure(figsize=(15, 5))
        plt.text(0.5, 0.5, 'No final metrics data available',
                horizontalalignment='center', verticalalignment='center')
        plt.tight_layout()
        return
        
    # Create subplots for each metric
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Training steps
    steps = [m['training_steps'] for m in metrics]
    x = np.arange(len(run_labels))
    axes[0].bar(x, steps)
    axes[0].set_title('Total Training Steps')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(run_labels, rotation=45, ha='right')
    
    # Final loss
    losses = [m['final_loss'] for m in metrics]
    axes[1].bar(x, losses)
    axes[1].set_title('Final Loss')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(run_labels, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('final_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_memory_usage():
    """Plot memory usage patterns if available in logs."""
    plt.figure(figsize=(12, 6))
    
    has_data = False
    for run_name, label in labels.items():
        if not os.path.exists(run_name):
            continue
            
        training_log, _ = load_run_data(run_name)
        
        # Check if memory usage data exists
        if training_log and any('memory_allocated' in log for log in training_log):
            steps = []
            memory = []
            for log in training_log:
                if 'memory_allocated' in log and log['memory_allocated'] is not None:
                    steps.append(log['step'])
                    memory.append(log['memory_allocated'] / 1e9)  # Convert to GB
            if steps and memory:
                plt.plot(steps, memory, label=label, alpha=0.8)
                has_data = True
    
    if not has_data:
        plt.text(0.5, 0.5, 'No memory usage data available',
                horizontalalignment='center', verticalalignment='center')
    
    plt.xlabel('Training Steps')
    plt.ylabel('GPU Memory Usage (GB)')
    plt.title('Memory Usage Across Runs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('memory_usage.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Set style for better visualization
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create plots
    plot_training_curves()
    plot_final_metrics()
    plot_memory_usage()
    
    print("Plots have been generated:")
    print("- training_loss.png: Training loss curves")
    print("- final_metrics.png: Comparison of final metrics")
    print("- memory_usage.png: Memory usage patterns")
