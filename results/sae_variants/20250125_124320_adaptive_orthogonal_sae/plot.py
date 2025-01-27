import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Define labels for the runs we want to plot
labels = {
    "run_1": "Fixed τ Top-k",
    "run_2": "Adaptive τ",
    "run_3": "Dynamic Pairs",
    "run_4": "Correlation Pruning",
    "run_5": "Feature Importance",
    "run_6": "Adaptive Sparsity",
    "run_7": "Local Competition",
    "run_8": "Adaptive Neighborhoods",
    "run_9": "Feature Reallocation"
}

def load_results(run_dir):
    """Load results from a run directory."""
    results_path = os.path.join(run_dir, "all_results.npy")
    if os.path.exists(results_path):
        with open(results_path, 'rb') as f:
            return np.load(f, allow_pickle=True).item()
    return None

def plot_training_curves():
    """Plot training curves for all runs."""
    plt.figure(figsize=(12, 8))
    
    for run_name, label in labels.items():
        results = load_results(run_name)
        if results and 'training_log' in results:
            losses = [log.get('loss', float('nan')) for log in results['training_log']]
            steps = range(len(losses))
            plt.plot(steps, losses, label=label, alpha=0.7)
    
    plt.xlabel('Training Steps')
    plt.ylabel('Total Loss')
    plt.title('Training Loss Curves')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

def plot_sparsity_evolution():
    """Plot sparsity levels over training."""
    plt.figure(figsize=(12, 8))
    
    for run_name, label in labels.items():
        results = load_results(run_name)
        if results and 'training_log' in results:
            sparsity = [log.get('sparsity_loss', float('nan')) for log in results['training_log']]
            steps = range(len(sparsity))
            plt.plot(steps, sparsity, label=label, alpha=0.7)
    
    plt.xlabel('Training Steps')
    plt.ylabel('Sparsity Level')
    plt.title('Feature Sparsity Evolution')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('sparsity_evolution.png')
    plt.close()

def plot_feature_correlations():
    """Plot average feature correlations."""
    correlations = []
    run_names = []
    
    for run_name, label in labels.items():
        results = load_results(run_name)
        if results and 'final_info' in results:
            # Extract final feature correlations if available
            if 'feature_correlations' in results['final_info']:
                correlations.append(results['final_info']['feature_correlations'])
                run_names.append(label)
    
    if correlations:
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(correlations)), correlations)
        plt.xticks(range(len(correlations)), run_names, rotation=45, ha='right')
        plt.ylabel('Average Feature Correlation')
        plt.title('Final Feature Correlations Across Runs')
        plt.tight_layout()
        plt.savefig('feature_correlations.png')
        plt.close()

def plot_feature_usage():
    """Plot feature usage distribution."""
    plt.figure(figsize=(12, 8))
    
    for run_name, label in labels.items():
        results = load_results(run_name)
        if results and 'final_info' in results:
            if 'feature_usage' in results['final_info']:
                usage = np.sort(results['final_info']['feature_usage'])[::-1]
                plt.plot(range(len(usage)), usage, label=label, alpha=0.7)
    
    plt.xlabel('Feature Index (sorted)')
    plt.ylabel('Usage Frequency')
    plt.title('Feature Usage Distribution')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('feature_usage.png')
    plt.close()

def main():
    """Generate all plots."""
    os.makedirs('plots', exist_ok=True)
    
    # Generate individual plots
    plot_training_curves()
    plot_sparsity_evolution()
    plot_feature_correlations()
    plot_feature_usage()
    
    print("Plots generated successfully!")

if __name__ == "__main__":
    main()
