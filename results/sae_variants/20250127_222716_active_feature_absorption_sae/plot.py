import matplotlib.pyplot as plt
import numpy as np

def plot_metrics_comparison(run_metrics, save_path=None):
    """Plot comparison of key metrics across different runs.
    
    Args:
        run_metrics: Dict mapping run names to their metrics
        save_path: Optional path to save the plot
    """
    metrics = ['cosine_similarity', 'mse', 'explained_variance', 
               'kl_div_score', 'mean_absorption_score']
    
    runs = list(run_metrics.keys())
    x = np.arange(len(metrics))
    width = 0.8 / len(runs)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, (run, metrics_dict) in enumerate(run_metrics.items()):
        values = [metrics_dict[m] for m in metrics]
        ax.bar(x + i*width - width*len(runs)/2, values, width, label=run)
    
    ax.set_ylabel('Score')
    ax.set_title('Comparison of Key Metrics Across Runs')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45)
    ax.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_absorption_distribution(absorption_rates, run_name, save_path=None):
    """Plot distribution of absorption rates across features.
    
    Args:
        absorption_rates: List of absorption rates
        run_name: Name of the run for the title
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.hist(absorption_rates, bins=30, alpha=0.75)
    plt.xlabel('Absorption Rate')
    plt.ylabel('Count')
    plt.title(f'Distribution of Absorption Rates - {run_name}')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_feature_activation_patterns(activations, run_name, save_path=None):
    """Plot heatmap of feature activation patterns.
    
    Args:
        activations: 2D array of feature activations
        run_name: Name of the run for the title
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 8))
    plt.imshow(activations, aspect='auto', cmap='viridis')
    plt.colorbar(label='Activation Strength')
    plt.xlabel('Feature Index')
    plt.ylabel('Sample Index')
    plt.title(f'Feature Activation Patterns - {run_name}')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
