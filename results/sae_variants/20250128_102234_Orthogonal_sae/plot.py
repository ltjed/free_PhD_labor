import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Style configuration
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.facecolor'] = '#f0f0f0'
sns.set_palette("husl")

# Dictionary mapping run numbers to learning rates and descriptions
labels = {
    "run_3": {"lr": 3e-4, "desc": "Optimized Sparsity"},
    "run_4": {"lr": 3e-4, "desc": "Enhanced Orthogonality"},
    "run_5": {"lr": 5e-4, "desc": "Optimized Learning Rate"},
    "run_6": {"lr": 7e-4, "desc": "Further LR Optimization"},
    "run_7": {"lr": 9e-4, "desc": "Final LR Optimization"},
    "run_8": {"lr": 1.1e-3, "desc": "Final LR Verification"}
}

def load_results(run_dir):
    """Load results from a run directory."""
    try:
        with open(Path(run_dir) / "final_info.json", 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"No results found in {run_dir}")
        return None

def extract_metrics(results):
    """Extract relevant metrics from results dictionary."""
    if not results or "training results for layer 19" not in results:
        return None
    
    core_results = results.get("core evaluation results", {}).get("metrics", {})
    
    return {
        "learning_rate": results["training results for layer 19"]["final_info"]["learning_rate"],
        "explained_variance": core_results.get("reconstruction_quality", {}).get("explained_variance", 0),
        "cosine_similarity": core_results.get("reconstruction_quality", {}).get("cossim", 0),
        "kl_div_score": core_results.get("model_behavior_preservation", {}).get("kl_div_score", 0),
        "feature_utilization": core_results.get("sparsity", {}).get("l0", 0) / 2304 * 100  # Convert to percentage
    }

def plot_metrics():
    """Generate plots for all metrics."""
    metrics_data = {}
    
    # Load data from all runs
    for run_name in labels.keys():
        results = load_results(run_name)
        if results:
            metrics = extract_metrics(results)
            if metrics:
                metrics_data[run_name] = metrics

    if not metrics_data:
        print("No valid data found to plot")
        return

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('SAE Performance Metrics Across Learning Rates', fontsize=16)

    # Learning rates for x-axis
    learning_rates = [metrics_data[run]["learning_rate"] for run in metrics_data]
    
    # Plot 1: Reconstruction Quality
    ax1.plot(learning_rates, [metrics_data[run]["explained_variance"] for run in metrics_data], 
             marker='o', label='Explained Variance')
    ax1.plot(learning_rates, [metrics_data[run]["cosine_similarity"] for run in metrics_data], 
             marker='s', label='Cosine Similarity')
    ax1.set_xlabel('Learning Rate')
    ax1.set_ylabel('Score')
    ax1.set_title('Reconstruction Quality')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Feature Utilization
    ax2.plot(learning_rates, [metrics_data[run]["feature_utilization"] for run in metrics_data], 
             marker='o', color='green')
    ax2.set_xlabel('Learning Rate')
    ax2.set_ylabel('Feature Utilization (%)')
    ax2.set_title('Feature Utilization')
    ax2.grid(True)

    # Plot 3: Model Behavior Preservation
    ax3.plot(learning_rates, [metrics_data[run]["kl_div_score"] for run in metrics_data], 
             marker='o', color='purple')
    ax3.set_xlabel('Learning Rate')
    ax3.set_ylabel('KL Divergence Score')
    ax3.set_title('Model Behavior Preservation')
    ax3.grid(True)

    # Plot 4: Combined Performance
    combined_performance = [
        (metrics_data[run]["explained_variance"] + 
         metrics_data[run]["cosine_similarity"] + 
         metrics_data[run]["kl_div_score"]) / 3 
        for run in metrics_data
    ]
    ax4.plot(learning_rates, combined_performance, marker='o', color='red')
    ax4.set_xlabel('Learning Rate')
    ax4.set_ylabel('Combined Score')
    ax4.set_title('Combined Performance Metric')
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig('sae_performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create a summary plot
    plt.figure(figsize=(12, 6))
    runs = list(metrics_data.keys())
    x = np.arange(len(runs))
    width = 0.2

    metrics = ['explained_variance', 'cosine_similarity', 'kl_div_score', 'feature_utilization']
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']

    for i, metric in enumerate(metrics):
        values = [metrics_data[run][metric] for run in runs]
        plt.bar(x + i*width, values, width, label=metric.replace('_', ' ').title(), color=colors[i])

    plt.xlabel('Run')
    plt.ylabel('Score')
    plt.title('Summary of Key Metrics Across Runs')
    plt.xticks(x + width*1.5, [f"{run}\n(lr={labels[run]['lr']})" for run in runs], rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('sae_metrics_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_metrics()
