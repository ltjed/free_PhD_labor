import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

def plot_condition_numbers(results_dir):
    """Plot condition numbers of feature subspaces across runs"""
    condition_numbers = []
    run_names = []
    
    # Load results from each run directory
    for d in sorted(os.listdir(results_dir)):
        if d.startswith('run_'):
            with open(os.path.join(results_dir, d, 'final_info.json')) as f:
                data = json.load(f)
                if 'condition_numbers' in data:
                    condition_numbers.append(data['condition_numbers'])
                    run_names.append(d)
    
    if not condition_numbers:
        print("No condition number data found")
        return
        
    # Create plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=condition_numbers)
    plt.xticks(range(len(run_names)), run_names, rotation=45)
    plt.ylabel('Condition Number')
    plt.title('Feature Subspace Condition Numbers Across Runs')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'condition_numbers.png'))
    plt.close()

def plot_sharing_impact(results_dir):
    """Plot impact of different alpha values on metrics"""
    alphas = []
    metrics = {'unlearning': [], 'reconstruction': [], 'absorption': []}
    
    # Load results
    for d in sorted(os.listdir(results_dir)):
        if d.startswith('run_'):
            with open(os.path.join(results_dir, d, 'final_info.json')) as f:
                data = json.load(f)
                if 'alpha' in data:
                    alphas.append(data['alpha'])
                    if 'unlearning evaluation results' in data:
                        metrics['unlearning'].append(
                            data['unlearning evaluation results']['eval_result_metrics']['unlearning']['unlearning_score']
                        )
                    if 'core evaluation results' in data:
                        metrics['reconstruction'].append(
                            data['core evaluation results']['metrics']['reconstruction_quality']['explained_variance']
                        )
                    if 'absorption evaluation results' in data:
                        metrics['absorption'].append(
                            data['absorption evaluation results']['eval_result_metrics']['mean']['mean_absorption_score']
                        )
    
    if not alphas:
        print("No alpha value data found")
        return
        
    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (metric, values) in enumerate(metrics.items()):
        if values:
            axes[i].scatter(alphas, values)
            axes[i].set_xlabel('Alpha Value')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].set_title(f'{metric.capitalize()} vs Alpha')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'alpha_impact.png'))
    plt.close()
