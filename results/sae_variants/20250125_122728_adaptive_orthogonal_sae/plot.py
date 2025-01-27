import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path

# Dictionary mapping run numbers to descriptive labels
labels = {
    "run_0": "Baseline",
    "run_3": "Adaptive τ",
    "run_6": "Hybrid Correlation-EMA",
    "run_9": "Dynamic Thresholding",
    "run_10": "Hierarchical Clustering"
}

def load_run_data(run_dir):
    """Load metrics from a run directory."""
    metrics_path = os.path.join(run_dir, "final_info.json")
    if not os.path.exists(metrics_path):
        return None
    
    with open(metrics_path, 'r') as f:
        return json.load(f)

def plot_comparison():
    """Generate comparison plots for different runs."""
    # Set up the plot with default style
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Collect data from each run
    runs = []
    losses = []
    accuracies = []
    
    for run_name, label in labels.items():
        data = load_run_data(run_name)
        if data:
            # Only append if we have valid numeric data
            final_loss = data.get('final_loss')
            test_accuracy = data.get('sae_test_accuracy')
            
            if final_loss is not None and test_accuracy is not None:
                runs.append(label)
                losses.append(float(final_loss))
                accuracies.append(float(test_accuracy))
            else:
                print(f"Warning: Skipping {label} due to missing metrics")
    
    if not runs:
        print("Error: No valid run data found with complete metrics")
        return
    
    # Convert to numpy arrays and handle any remaining None values
    losses = np.array(losses, dtype=np.float32)
    accuracies = np.array(accuracies, dtype=np.float32)
    
    # Replace any remaining NaN/inf values with 0
    losses = np.nan_to_num(losses, nan=0.0, posinf=0.0, neginf=0.0)
    accuracies = np.nan_to_num(accuracies, nan=0.0, posinf=0.0, neginf=0.0)
        
    # Plot final losses
    x = np.arange(len(runs))
    ax1.bar(x, losses, alpha=0.8, color='skyblue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(runs, rotation=45, ha='right')
    ax1.set_ylabel('Final Loss')
    ax1.set_title('Comparison of Final Losses')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot accuracies
    ax2.bar(x, accuracies, alpha=0.8, color='lightgreen')
    ax2.set_xticks(x)
    ax2.set_xticklabels(runs, rotation=45, ha='right')
    ax2.set_ylabel('Test Accuracy')
    ax2.set_title('Comparison of Test Accuracies')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('comparison_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_correlations():
    """Plot feature correlation heatmaps for selected runs."""
    valid_runs = []
    for run_name, label in labels.items():
        correlation_path = os.path.join(run_name, "feature_correlations.npy")
        if os.path.exists(correlation_path):
            valid_runs.append((run_name, label))
    
    if not valid_runs:
        print("Warning: No correlation data found")
        return
        
    fig, axes = plt.subplots(1, len(valid_runs), figsize=(20, 4))
    if len(valid_runs) == 1:
        axes = [axes]
    
    for idx, (run_name, label) in enumerate(valid_runs):
        correlation_path = os.path.join(run_name, "feature_correlations.npy")
        correlations = np.load(correlation_path)
        im = axes[idx].imshow(correlations, cmap='coolwarm', vmin=-1, vmax=1)
        axes[idx].set_title(label)
        axes[idx].axis('off')
    
    plt.colorbar(im, ax=axes.ravel().tolist())
    plt.tight_layout()
    plt.savefig('correlation_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Create plots
    plot_comparison()
    plot_feature_correlations()
    
    print("Plots have been generated:")
    print("1. comparison_plots.png - Shows final losses and accuracies")
    print("2. correlation_heatmaps.png - Shows feature correlation patterns")
