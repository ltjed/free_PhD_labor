import matplotlib.pyplot as plt
import numpy as np
import json
import os

def load_results(run_dir):
    """Load results from a run directory"""
    with open(os.path.join(run_dir, "final_info.json"), "r") as f:
        return json.load(f)

def plot_training_curves(results_dict, out_dir):
    """Plot training curves comparing different runs"""
    plt.figure(figsize=(12, 6))
    
    # Dictionary mapping run names to display labels
    labels = {
        "run_0": "Baseline",
        "run_9": "Mixed Precision",
        "run_10": "Stabilized Training"
    }
    
    for run_name, label in labels.items():
        if run_name in results_dict:
            run_data = results_dict[run_name]
            training_results = run_data.get("training results", {})
            if training_results:
                steps = training_results.get("final_info", {}).get("training_steps", 0)
                final_loss = training_results.get("final_info", {}).get("final_loss", None)
                if final_loss is not None:
                    plt.scatter(steps, final_loss, label=f"{label} (Loss: {final_loss:.2f})")
    
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss Comparison Across Runs")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "training_curves.png"))
    plt.close()

def plot_evaluation_metrics(results_dict, out_dir):
    """Plot evaluation metrics comparison"""
    metrics = ["kl_div_score", "ce_loss_score", "explained_variance"]
    
    labels = {
        "run_0": "Baseline",
        "run_9": "Mixed Precision",
        "run_10": "Stabilized Training"
    }
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
    
    for i, metric in enumerate(metrics):
        values = []
        names = []
        
        for run_name, label in labels.items():
            if run_name in results_dict:
                run_data = results_dict[run_name]
                core_results = run_data.get("core evaluation results", {})
                if core_results:
                    metric_value = core_results.get("metrics", {}).get("model_behavior_preservation", {}).get(metric)
                    if metric_value is not None:
                        values.append(metric_value)
                        names.append(label)
        
        if values:
            axes[i].bar(names, values)
            axes[i].set_title(metric.replace("_", " ").title())
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "evaluation_metrics.png"))
    plt.close()

def plot_sparsity_analysis(results_dict, out_dir):
    """Plot sparsity-related metrics"""
    plt.figure(figsize=(10, 6))
    
    labels = {
        "run_0": "Baseline",
        "run_9": "Mixed Precision",
        "run_10": "Stabilized Training"
    }
    
    l0_values = []
    l1_values = []
    names = []
    
    for run_name, label in labels.items():
        if run_name in results_dict:
            run_data = results_dict[run_name]
            core_results = run_data.get("core evaluation results", {})
            if core_results:
                metrics = core_results.get("metrics", {}).get("sparsity", {})
                if metrics:
                    l0_values.append(metrics.get("l0", 0))
                    l1_values.append(metrics.get("l1", 0))
                    names.append(label)
    
    x = np.arange(len(names))
    width = 0.35
    
    plt.bar(x - width/2, l0_values, width, label='L0 Sparsity')
    plt.bar(x + width/2, l1_values, width, label='L1 Sparsity')
    
    plt.xlabel("Run")
    plt.ylabel("Sparsity Value")
    plt.title("Sparsity Metrics Comparison")
    plt.xticks(x, names, rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sparsity_analysis.png"))
    plt.close()

if __name__ == "__main__":
    # Create plots directory
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load results from different runs
    results = {}
    run_dirs = ["run_0", "run_9", "run_10"]
    
    for run_dir in run_dirs:
        if os.path.exists(run_dir):
            results[run_dir] = load_results(run_dir)
    
    # Generate plots
    plot_training_curves(results, plots_dir)
    plot_evaluation_metrics(results, plots_dir)
    plot_sparsity_analysis(results, plots_dir)
