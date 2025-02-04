import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Dictionary mapping run names to their directories
labels = {
    "Baseline": "run_0",
    "Initial Competition": "run_1", 
    "Competition + Warmup": "run_2",
    "Enhanced Competition": "run_3",
    "Balanced Competition": "run_4",
    "Enhanced Separation": "run_5",
    "Final Tuned": "run_6"
}

def load_results(run_dir):
    """Load results from a run directory"""
    path = Path(run_dir) / "final_info.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

def plot_absorption_scores():
    """Plot absorption scores across runs"""
    scores = []
    names = []
    
    for name, run_dir in labels.items():
        results = load_results(run_dir)
        if results and 'absorption evaluation results' in results:
            score = results['absorption evaluation results']['eval_result_metrics']['mean']['mean_absorption_score']
            scores.append(score)
            names.append(name)
    
    plt.figure(figsize=(10, 6))
    plt.bar(names, scores)
    plt.title('Absorption Scores Across Runs (Lower is Better)')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Absorption Score')
    plt.tight_layout()
    plt.savefig('absorption_scores.png')
    plt.close()

def plot_core_metrics():
    """Plot core metrics across runs"""
    metrics = {
        'KL Divergence': [],
        'MSE': [],
        'Explained Variance': []
    }
    names = []
    
    for name, run_dir in labels.items():
        results = load_results(run_dir)
        if results and 'core evaluation results' in results:
            core = results['core evaluation results']['metrics']
            metrics['KL Divergence'].append(core['model_behavior_preservation']['kl_div_score'])
            metrics['MSE'].append(core['reconstruction_quality']['mse'])
            metrics['Explained Variance'].append(core['reconstruction_quality']['explained_variance'])
            names.append(name)
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(names))
    width = 0.25
    
    plt.bar(x - width, metrics['KL Divergence'], width, label='KL Divergence')
    plt.bar(x, metrics['MSE'], width, label='MSE')
    plt.bar(x + width, metrics['Explained Variance'], width, label='Explained Variance')
    
    plt.title('Core Metrics Across Runs')
    plt.xticks(x, names, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('core_metrics.png')
    plt.close()

def plot_sparse_probing():
    """Plot sparse probing accuracy across runs"""
    accuracies = []
    names = []
    
    for name, run_dir in labels.items():
        results = load_results(run_dir)
        if results and 'sparse probing evaluation results' in results:
            acc = results['sparse probing evaluation results']['eval_result_metrics']['sae']['sae_test_accuracy']
            accuracies.append(acc)
            names.append(name)
    
    plt.figure(figsize=(10, 6))
    plt.bar(names, accuracies)
    plt.title('Sparse Probing Accuracy Across Runs')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig('sparse_probing.png')
    plt.close()

def plot_scr_metrics():
    """Plot SCR metrics across runs"""
    thresholds = [2, 5, 10, 20, 50, 100]
    metrics = {t: [] for t in thresholds}
    names = []
    
    for name, run_dir in labels.items():
        results = load_results(run_dir)
        if results and 'scr and tpp evaluations results' in results:
            eval_metrics = results['scr and tpp evaluations results']['eval_result_metrics']
            # Handle both possible metric locations
            scr = eval_metrics.get('scr_metrics', eval_metrics)
            try:
                has_metrics = False
                for t in thresholds:
                    key = f'scr_metric_threshold_{t}'
                    if key in scr:
                        metrics[t].append(scr[key])
                        has_metrics = True
                if has_metrics:
                    names.append(name)
            except (KeyError, TypeError):
                print(f"Warning: Could not extract SCR metrics for {name}")
        else:
            print(f"Warning: No SCR results found for {name}")
    
    if not names:
        print("No valid SCR metrics found in any run")
        return
        
    plt.figure(figsize=(12, 6))
    for t in thresholds:
        if metrics[t]:  # Only plot if we have data
            plt.plot(names, metrics[t], marker='o', label=f'Threshold {t}')
    
    plt.title('SCR Metrics Across Runs')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('SCR Score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('scr_metrics.png')
    plt.close()

def main():
    # Set default style parameters
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    
    # Generate all plots
    plot_absorption_scores()
    plot_core_metrics() 
    plot_sparse_probing()
    plot_scr_metrics()

if __name__ == "__main__":
    main()
