import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import seaborn as sns

# Set style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
sns.set_palette("deep")

# Dictionary mapping run numbers to their labels
labels = {
    "Run 0": "Baseline SAE",
    "Run 1": "Initial CSAE (0.1)",
    "Run 2": "CSAE (0.3)",
    "Run 3": "CSAE + Contrastive",
    "Run 4": "CSAE + Gradient Penalty"
}

def load_run_data(run_dir):
    """Load final_info.json from a run directory."""
    info_path = os.path.join(run_dir, "final_info.json")
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            return json.load(f)
    return None

def plot_loss_comparison():
    """Plot final loss comparison across runs."""
    losses = []
    run_labels = []
    
    for run_name, label in labels.items():
        data = load_run_data(run_name)
        if data and 'final_loss' in data and data['final_loss'] is not None:
            losses.append(data['final_loss'])
            run_labels.append(label)
    
    if losses:
        plt.figure(figsize=(10, 6))
        plt.bar(run_labels, losses)
        plt.title('Final Loss Comparison Across Runs')
        plt.ylabel('Final Loss')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('loss_comparison.png')
        plt.close()

def plot_nces_evolution():
    """Plot NCES evolution over training steps."""
    plt.figure(figsize=(10, 6))
    
    for run_name, label in labels.items():
        data = load_run_data(run_name)
        if data and 'nces_history' in data:
            steps = range(len(data['nces_history']))
            plt.plot(steps, data['nces_history'], label=label)
    
    plt.title('NCES Evolution During Training')
    plt.xlabel('Training Steps')
    plt.ylabel('Normalized Causal Effect Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig('nces_evolution.png')
    plt.close()

def plot_feature_correlations(run_name="Run 4"):
    """Plot feature correlation matrix for the final run."""
    data = load_run_data(run_name)
    if data and 'feature_correlations' in data:
        correlations = np.array(data['feature_correlations'])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlations, cmap='coolwarm', center=0)
        plt.title(f'Feature Correlations ({labels[run_name]})')
        plt.tight_layout()
        plt.savefig('feature_correlations.png')
        plt.close()

def plot_intervention_distribution():
    """Plot distribution of intervention sizes across features."""
    plt.figure(figsize=(10, 6))
    
    for run_name, label in labels.items():
        data = load_run_data(run_name)
        if data and 'intervention_sizes' in data:
            sns.kdeplot(data['intervention_sizes'], label=label)
    
    plt.title('Distribution of Intervention Sizes')
    plt.xlabel('Intervention Size')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig('intervention_distribution.png')
    plt.close()

def main():
    """Generate all plots."""
    # Create output directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Generate plots
    plot_loss_comparison()
    plot_nces_evolution()
    plot_feature_correlations()
    plot_intervention_distribution()
    
    print("Plots generated successfully!")

if __name__ == "__main__":
    main()
