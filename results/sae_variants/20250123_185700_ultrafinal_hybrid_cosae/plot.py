import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Dictionary mapping run numbers to descriptive labels
labels = {
    "run_0": "Baseline",
    "run_1": "JL Projection + EMA",
    "run_2": "Dynamic Threshold",
    "run_3": "Adaptive λ Bounds", 
    "run_4": "Feature Diversity Loss",
    "run_5": "Gram-Schmidt Ortho",
    "run_6": "Gradient Balancing",
    "run_7": "Hierarchical Groups",
    "run_8": "Adaptive Learning",
    "run_9": "Winner-Take-All",
    "run_10": "Hard Orthogonality"
}

def load_run_results(run_dir):
    """Load results from a run directory."""
    info_path = os.path.join(run_dir, "final_info.json")
    if not os.path.exists(info_path):
        return None
    
    with open(info_path, 'r') as f:
        return json.load(f)

def plot_absorption_scores(results_dict):
    """Plot absorption scores across runs."""
    runs = []
    scores = []
    
    for run, data in results_dict.items():
        if data and 'eval_result_metrics' in data:
            if 'mean' in data['eval_result_metrics']:
                score = data['eval_result_metrics']['mean'].get('mean_absorption_score', 0)
                runs.append(labels.get(run, run))
                scores.append(score)
    
    if not runs:
        return
    
    plt.figure(figsize=(12, 6))
    plt.bar(runs, scores)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Mean Absorption Score')
    plt.title('Absorption Scores Across Runs')
    plt.tight_layout()
    plt.savefig('absorption_scores.png')
    plt.close()

def plot_feature_splits(results_dict):
    """Plot number of split features across runs."""
    runs = []
    splits = []
    
    for run, data in results_dict.items():
        if data and 'eval_result_metrics' in data:
            if 'mean' in data['eval_result_metrics']:
                split = data['eval_result_metrics']['mean'].get('mean_num_split_features', 0)
                runs.append(labels.get(run, run))
                splits.append(split)
    
    if not splits:
        return
    
    plt.figure(figsize=(12, 6))
    plt.bar(runs, splits)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Mean Number of Split Features')
    plt.title('Feature Splitting Across Runs')
    plt.tight_layout()
    plt.savefig('feature_splits.png')
    plt.close()

def plot_loss_comparison(results_dict):
    """Plot final loss values across runs."""
    runs = []
    losses = []
    
    for run, data in results_dict.items():
        if data and 'final_loss' in data and data['final_loss'] is not None:
            runs.append(labels.get(run, run))
            losses.append(data['final_loss'])
    
    if not losses:
        return
    
    plt.figure(figsize=(12, 6))
    plt.bar(runs, losses)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Final Loss')
    plt.title('Final Loss Values Across Runs')
    plt.tight_layout()
    plt.savefig('final_losses.png')
    plt.close()

def plot_letter_absorption(results_dict):
    """Plot absorption rates by first letter."""
    latest_run = None
    latest_data = None
    
    # Find the latest run with letter data
    for run, data in sorted(results_dict.items(), reverse=True):
        if data and 'eval_result_details' in data:
            details = data['eval_result_details']
            if details and 'first_letter' in details[0]:
                latest_run = run
                latest_data = details
                break
    
    if not latest_data:
        return
    
    letters = [d['first_letter'] for d in latest_data]
    rates = [d['absorption_rate'] for d in latest_data]
    
    plt.figure(figsize=(15, 6))
    plt.bar(letters, rates)
    plt.ylabel('Absorption Rate')
    plt.title(f'Letter-wise Absorption Rates ({labels.get(latest_run, latest_run)})')
    plt.tight_layout()
    plt.savefig('letter_absorption.png')
    plt.close()

def main():
    # Load results from all runs
    results = {}
    for run_name in labels.keys():
        run_dir = run_name
        if os.path.exists(run_dir):
            results[run_name] = load_run_results(run_dir)
    
    # Generate plots
    plot_absorption_scores(results)
    plot_feature_splits(results)
    plot_loss_comparison(results)
    plot_letter_absorption(results)

if __name__ == "__main__":
    main()
