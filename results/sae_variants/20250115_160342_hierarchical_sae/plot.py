import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Define which runs to plot and their labels
labels = {
    "run_0": "Baseline",
    "run_1": "Initial Hierarchical SAE",
    "run_2": "Flexible Composition Loss",
    "run_3": "Improved Init & Warmup",
    "run_4": "Gradient Scaling",
    "run_5": "Layer Norm + Tier Clipping"
}

# Metrics to plot
metrics = [
    'l2_loss',
    'sparsity_loss',
    'loss',
    'tier_activity_0',
    'tier_activity_1', 
    'tier_activity_2'
]

def load_run_data(run_dir):
    """Load training log from a run directory"""
    results_path = os.path.join(run_dir, "all_results.npy")
    if not os.path.exists(results_path):
        return None
        
    data = np.load(results_path, allow_pickle=True).item()
    return data['training_log']

def plot_metrics():
    """Plot metrics across runs"""
    plt.figure(figsize=(15, 10))
    
    # Create subplots for each metric
    for i, metric in enumerate(metrics):
        plt.subplot(2, 3, i+1)
        
        for run, label in labels.items():
            data = load_run_data(run)
            if data is None:
                continue
                
            # Extract metric values
            values = [step.get(metric, np.nan) for step in data]
            steps = range(len(values))
            
            # Plot with run-specific styling
            plt.plot(steps, values, label=label)
            plt.title(metric.replace('_', ' ').title())
            plt.xlabel('Training Steps')
            plt.grid(True)
            
        if i == 0:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_tier_activity_comparison():
    """Plot final tier activity comparison across runs"""
    final_activities = defaultdict(list)
    
    for run, label in labels.items():
        data = load_run_data(run)
        if data is None:
            continue
            
        # Get final tier activities
        for tier in range(3):
            metric = f'tier_activity_{tier}'
            values = [step.get(metric, np.nan) for step in data]
            final_activities[tier].append(values[-1] if values else np.nan)
    
    # Plot as grouped bar chart
    x = np.arange(len(labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for tier in range(3):
        ax.bar(x + tier*width, final_activities[tier], width, label=f'Tier {tier+1}')
    
    ax.set_xlabel('Run')
    ax.set_ylabel('Final Activity')
    ax.set_title('Final Tier Activity Comparison')
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels.values(), rotation=45, ha='right')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('tier_activity_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Create output directory for plots
    os.makedirs('plots', exist_ok=True)
    
    # Generate plots
    plot_metrics()
    plot_tier_activity_comparison()
    
    print("Plots saved to ./plots/ directory")

if __name__ == "__main__":
    main()
