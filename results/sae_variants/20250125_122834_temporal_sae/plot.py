import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from pathlib import Path

# Style configuration
plt.style.use('default')  # Use default style as base
# Configure style manually
plt.rcParams.update({
    'figure.figsize': [12, 8],
    'figure.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'lines.linewidth': 2,
    'axes.spines.top': False,
    'axes.spines.right': False,
})
# Use a colorblind-friendly color palette
colors = ['#0077BB', '#EE7733', '#009988', '#CC3311', '#33BBEE', '#EE3377']
sns.set_palette(colors)

# Labels for each run (only include runs we want to plot)
labels = {
    "run_9": "Hierarchical Architecture",
    "run_10": "Curriculum Learning"
}

def load_run_data(run_dir):
    """Load training data for a given run directory."""
    try:
        with open(os.path.join(run_dir, "final_info.json"), 'r') as f:
            info = json.load(f)
        return info
    except FileNotFoundError:
        return None

def plot_training_progression():
    """Plot training progression across runs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for run_name, label in labels.items():
        data = load_run_data(run_name)
        if data:
            # Plot training steps
            ax1.bar(label, data.get('training_steps', 0))
            
            # Plot final loss if available
            final_loss = data.get('final_loss')
            if final_loss is not None:
                ax2.bar(label, final_loss)
    
    ax1.set_title('Training Steps Completed')
    ax1.set_ylabel('Steps')
    ax1.tick_params(axis='x', rotation=45)
    
    ax2.set_title('Final Training Loss')
    ax2.set_ylabel('Loss')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('training_progression.png')
    plt.close()

def plot_hyperparameters():
    """Plot hyperparameter comparison across runs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for run_name, label in labels.items():
        data = load_run_data(run_name)
        if data:
            # Plot learning rate
            ax1.bar(label, data.get('learning_rate', 0))
            
            # Plot sparsity penalty
            ax2.bar(label, data.get('sparsity_penalty', 0))
    
    ax1.set_title('Learning Rate')
    ax1.set_ylabel('Rate')
    ax1.tick_params(axis='x', rotation=45)
    
    ax2.set_title('Sparsity Penalty')
    ax2.set_ylabel('Penalty')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('hyperparameters.png')
    plt.close()

def main():
    """Generate all plots."""
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Generate plots
    plot_training_progression()
    plot_hyperparameters()
    
    print("Plots have been generated in the plots directory.")

if __name__ == "__main__":
    main()
