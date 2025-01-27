import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Style configuration
plt.style.use('default')  # Use default style as base
plt.rcParams.update({
    'figure.figsize': [12, 8],
    'font.size': 12,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False
})
sns.set_palette("husl")  # Set color palette after style

# Dictionary mapping run names to display labels
labels = {
    'run_0': 'Baseline',
    'run_1': 'Fixed τ=0.2',
    'run_2': 'Fixed τ=0.05',
    'run_4': 'Adaptive τ + Constrained',
    'run_5': 'High LR + Adaptive τ'
}

def load_results(run_dir):
    """Load results from a run directory."""
    path = Path(run_dir) / "final_info.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

def plot_core_metrics():
    """Plot core metrics comparison across runs."""
    metrics = {
        'Reconstruction (cosine sim)': [],
        'Sparsity (L0)': [],
        'Model Preservation (KL div)': []
    }
    
    runs = []
    for run, label in labels.items():
        results = load_results(run)
        if results and 'core evaluation results' in results:
            core_metrics = results['core evaluation results']['metrics']
            metrics['Reconstruction (cosine sim)'].append(core_metrics['reconstruction_quality']['cossim'])
            metrics['Sparsity (L0)'].append(core_metrics['sparsity']['l0'])
            metrics['Model Preservation (KL div)'].append(core_metrics['model_behavior_preservation']['kl_div_score'])
            runs.append(label)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Core Metrics Comparison Across Runs', fontsize=14)
    
    for ax, (metric, values) in zip(axes, metrics.items()):
        ax.bar(runs, values)
        ax.set_title(metric)
        ax.tick_params(axis='x', rotation=45)
        if metric == 'Sparsity (L0)':
            ax.set_ylim(0, max(values) * 1.2)
    
    plt.tight_layout()
    plt.savefig('core_metrics_comparison.png')
    plt.close()

def plot_sparse_probing():
    """Plot sparse probing accuracy comparison."""
    llm_accuracies = []
    sae_accuracies = []
    runs = []
    
    for run, label in labels.items():
        results = load_results(run)
        if results and 'sparse probing evaluation results' in results:
            metrics = results['sparse probing evaluation results']['eval_result_metrics']
            llm_accuracies.append(metrics['llm']['llm_test_accuracy'])
            sae_accuracies.append(metrics['sae']['sae_test_accuracy'])
            runs.append(label)
    
    width = 0.35
    x = np.arange(len(runs))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, llm_accuracies, width, label='LLM')
    ax.bar(x + width/2, sae_accuracies, width, label='SAE')
    
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Sparse Probing Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(runs, rotation=45)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('sparse_probing_comparison.png')
    plt.close()

def plot_scr_metrics():
    """Plot SCR metrics across thresholds."""
    thresholds = [2, 5, 10, 20, 50]
    metrics_by_run = {}
    
    for run, label in labels.items():
        results = load_results(run)
        if results and 'scr and tpp evaluations results' in results:
            metrics = results['scr and tpp evaluations results']['eval_result_metrics']['scr_metrics']
            metrics_by_run[label] = [metrics[f'scr_metric_threshold_{t}'] for t in thresholds]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for label, values in metrics_by_run.items():
        ax.plot(thresholds, values, marker='o', label=label)
    
    ax.set_xlabel('Threshold')
    ax.set_ylabel('SCR Metric')
    ax.set_title('SCR Metrics Across Thresholds')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('scr_metrics_comparison.png')
    plt.close()

def main():
    """Generate all plots."""
    plot_core_metrics()
    plot_sparse_probing()
    plot_scr_metrics()

if __name__ == "__main__":
    main()
