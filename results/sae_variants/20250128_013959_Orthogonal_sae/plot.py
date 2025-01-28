import matplotlib.pyplot as plt
import numpy as np
import json
import os

# Dictionary mapping run numbers to their descriptive labels
labels = {
    "run_0": "Baseline",
    "run_2": "Ortho (α=0.0, p=1.0)", 
    "run_3": "Ortho (α=0.0, p=2.0)",
    "run_4": "Ortho (α=0.0, p=4.0)",
    "run_5": "Ortho (α=0.1, p=2.0)"
}

# Metrics to plot
metrics = {
    "final_loss": "Final Loss",
    "kl_div": "KL Divergence",
    "mse": "MSE",
    "l0_sparsity": "L0 Sparsity"
}

def load_run_data(run_dir):
    """Load results from a run directory."""
    try:
        with open(os.path.join(run_dir, "final_info.json"), "r") as f:
            data = json.load(f)
            layer_data = data["training results for layer 19"]
            core_data = data.get("core evaluation results", {})
            
            # Extract metrics
            return {
                "final_loss": layer_data["final_info"]["final_loss"],
                "kl_div": core_data.get("metrics", {}).get("model_behavior_preservation", {}).get("kl_div_score", None),
                "mse": core_data.get("metrics", {}).get("reconstruction_quality", {}).get("mse", None),
                "l0_sparsity": core_data.get("metrics", {}).get("sparsity", {}).get("l0", None)
            }
    except FileNotFoundError:
        return None

def create_plots():
    """Create plots comparing metrics across runs."""
    # Collect data
    data = {}
    for run_name in labels.keys():
        run_data = load_run_data(run_name)
        if run_data:
            data[run_name] = run_data

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Orthogonal SAE Performance Metrics Across Runs', fontsize=16)
    
    # Plot each metric
    for (metric, title), ax in zip(metrics.items(), axes.flat):
        values = [data[run][metric] for run in data if data[run][metric] is not None]
        x = range(len(values))
        run_labels = [labels[run] for run in data if data[run][metric] is not None]
        
        bars = ax.bar(x, values)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(run_labels, rotation=45, ha='right')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
    
    # Create trend plot
    plt.figure(figsize=(12, 6))
    for metric in metrics:
        values = [data[run][metric] for run in data if data[run][metric] is not None]
        # Normalize values to show relative changes
        normalized = np.array(values) / values[0]
        plt.plot(range(len(values)), normalized, marker='o', label=metrics[metric])
    
    plt.title('Relative Improvement Over Baseline')
    plt.xlabel('Run')
    plt.xticks(range(len(data)), [labels[run] for run in data], rotation=45, ha='right')
    plt.ylabel('Relative Change (Baseline = 1.0)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('relative_improvements.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    create_plots()
