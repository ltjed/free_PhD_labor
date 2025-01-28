import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Dictionary mapping run names to their display labels
labels = {
    "Run 0": "Baseline TopK",
    "Run 1": "Linear Sparsity",
    "Run 2": "Feature Resampling", 
    "Run 4": "Adaptive Sparsity",
    "Run 5": "Feature Clustering"
}

def load_results(run_dir):
    """Load results from final_info.json for a given run directory"""
    try:
        with open(Path(run_dir) / "final_info.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"No results found for {run_dir}")
        return None

def plot_metrics_comparison():
    """Plot key metrics comparison across runs"""
    metrics = {
        "Explained Variance": [],
        "KL Divergence": [],
        "Active Features": [],
        "L1 Sparsity": []
    }
    
    runs = []
    for run_name in labels.keys():
        results = load_results(run_name)
        if results:
            runs.append(labels[run_name])
            core_results = results.get("core evaluation results", {}).get("metrics", {})
            
            # Extract metrics with proper nesting
            recon_quality = core_results.get("reconstruction_quality", {})
            model_behavior = core_results.get("model_behavior_preservation", {})
            sparsity = core_results.get("sparsity", {})
            
            metrics["Explained Variance"].append(recon_quality.get("explained_variance", 0))
            metrics["KL Divergence"].append(model_behavior.get("kl_div_with_sae", 0))
            metrics["Active Features"].append(sparsity.get("l0", 0))
            metrics["L1 Sparsity"].append(sparsity.get("l1", 0))

    # Create subplots for each metric
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Performance Metrics Across Different Runs", fontsize=16, y=1.05)
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f1c40f']
    for (metric, values), ax, color in zip(metrics.items(), axes.flat, colors):
        bars = ax.bar(runs, values, color=color)
        ax.set_title(metric, pad=10)
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('metrics_comparison.png')
    plt.close()

def plot_absorption_rates():
    """Plot absorption rates across runs"""
    absorption_rates = []
    runs = []
    
    for run_name in labels.keys():
        results = load_results(run_name)
        if results:
            runs.append(labels[run_name])
            absorption_results = results.get("absorption evaluation results", {})
            mean_absorption = absorption_results.get("eval_result_metrics", {}).get("mean", {}).get("mean_absorption_score", 0)
            absorption_rates.append(mean_absorption)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(runs, absorption_rates, color='#3498db')
    plt.title("Feature Absorption Rates Across Runs", pad=20)
    plt.ylabel("Mean Absorption Score")
    plt.xticks(rotation=45)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('absorption_rates.png')
    plt.close()

def plot_probing_accuracy():
    """Plot sparse probing accuracy across runs"""
    accuracies = {
        "Top-1": [],
        "Top-5": [],
        "Top-20": []
    }
    runs = []
    
    for run_name in labels.keys():
        results = load_results(run_name)
        if results:
            runs.append(labels[run_name])
            probing_results = results.get("sparse probing evaluation results", {}).get("eval_result_metrics", {}).get("sae", {})
            
            accuracies["Top-1"].append(probing_results.get("sae_top_1_test_accuracy", 0))
            accuracies["Top-5"].append(probing_results.get("sae_top_5_test_accuracy", 0))
            accuracies["Top-20"].append(probing_results.get("sae_top_20_test_accuracy", 0))
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(runs))
    width = 0.25
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    for i, ((k, v), color) in enumerate(zip(accuracies.items(), colors)):
        bars = plt.bar(x + i*width, v, width, label=k, color=color)
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')
    
    plt.xlabel("Runs")
    plt.ylabel("Accuracy")
    plt.title("Sparse Probing Accuracy Across Runs", pad=20)
    plt.xticks(x + width, runs, rotation=45)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('probing_accuracy.png')
    plt.close()

def main():
    # Set basic style parameters
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    
    # Create all plots
    plot_metrics_comparison()
    plot_absorption_rates()
    plot_probing_accuracy()

if __name__ == "__main__":
    main()
