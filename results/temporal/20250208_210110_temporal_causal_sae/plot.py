import matplotlib.pyplot as plt
import numpy as np
import json
import os

def load_results(run_dir):
    with open(os.path.join(run_dir, "final_info.json"), "r") as f:
        return json.load(f)

def plot_metrics_comparison(run_dirs, labels):
    metrics = {
        'absorption_score': [],
        'unlearning_score': [],
        'sparsity_l0': [],
        'probing_top1': [],
        'training_loss': []
    }
    
    for run_dir in run_dirs:
        results = load_results(run_dir)
        layer_results = results.get("training results for layer 12", {})
        
        # Extract metrics
        metrics['absorption_score'].append(
            results.get("absorption evaluation results", {})
            .get("eval_result_metrics", {})
            .get("mean", {})
            .get("mean_absorption_score", 0)
        )
        
        metrics['unlearning_score'].append(
            results.get("unlearning evaluation results", {})
            .get("eval_result_metrics", {})
            .get("unlearning", {})
            .get("unlearning_score", 0)
        )
        
        metrics['sparsity_l0'].append(
            results.get("core evaluation results", {})
            .get("metrics", {})
            .get("sparsity", {})
            .get("l0", 0)
        )
        
        metrics['probing_top1'].append(
            results.get("sparse probing evaluation results", {})
            .get("eval_result_metrics", {})
            .get("sae", {})
            .get("sae_top_1_test_accuracy", 0)
        )
        
        metrics['training_loss'].append(
            layer_results.get("final_info", {}).get("final_loss", 0)
        )

    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Metrics Comparison Across Runs')
    
    for idx, (metric, values) in enumerate(metrics.items()):
        row = idx // 3
        col = idx % 3
        if idx < 5:  # Skip the empty 6th subplot
            axes[row, col].bar(labels, values)
            axes[row, col].set_title(metric.replace('_', ' ').title())
            axes[row, col].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('metrics_comparison.png')
    plt.close()

if __name__ == "__main__":
    run_dirs = ["run_0", "run_1"]  # Add more runs as needed
    labels = ["Baseline", "Temporal"]  # Add corresponding labels
    plot_metrics_comparison(run_dirs, labels)
