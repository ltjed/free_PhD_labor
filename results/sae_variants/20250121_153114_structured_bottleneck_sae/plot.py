import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import pandas as pd

# Configure plot style
plt.style.use('bmh')  # Using a built-in matplotlib style
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])

# Define labels for each run
labels = {
    'run_0': 'Baseline',
    'run_2': 'Optimized Groups',
    'run_3': 'Enhanced Isolation',
    'run_4': 'Aggressive Separation',
    'run_5': 'Feature Disentanglement'
}

def load_run_data(run_dir):
    """Load results from a run directory."""
    try:
        with open(Path(run_dir) / "final_info.json") as f:
            info = json.load(f)
        with open(Path(run_dir) / "all_results.npy", 'rb') as f:
            results = np.load(f, allow_pickle=True).item()
        return info, results
    except FileNotFoundError:
        return None, None

def plot_unlearning_scores():
    """Plot unlearning scores across runs."""
    scores = []
    runs = []
    
    for run, label in labels.items():
        try:
            info, _ = load_run_data(run)
            if info and 'eval_result_metrics' in info:
                if 'unlearning' in info['eval_result_metrics']:
                    score = info['eval_result_metrics']['unlearning']['unlearning_score']
                    scores.append(score)
                    runs.append(label)
        except Exception as e:
            print(f"Warning: Could not process unlearning score for {label}: {str(e)}")
    
    if scores:
        try:
            plt.figure()
            plt.bar(runs, scores)
            plt.title('Unlearning Scores Across Runs')
            plt.ylabel('Unlearning Score')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('unlearning_scores.png')
            print(f"Successfully saved unlearning scores plot with {len(runs)} runs")
        except Exception as e:
            print(f"Error creating unlearning scores plot: {str(e)}")
    else:
        print("No valid unlearning scores found for plotting")

def plot_training_curves():
    """Plot training loss curves for all runs."""
    has_data = False
    
    try:
        plt.figure()
        
        for run, label in labels.items():
            try:
                _, results = load_run_data(run)
                if results and 'training_log' in results and results['training_log']:
                    losses = [log.get('loss', float('nan')) for log in results['training_log']]
                    if any(not np.isnan(loss) for loss in losses):  # Check if we have valid losses
                        steps = range(len(losses))
                        plt.plot(steps, losses, label=label)
                        has_data = True
            except Exception as e:
                print(f"Warning: Could not process training curves for {label}: {str(e)}")
        
        if has_data:
            plt.title('Training Loss Curves')
            plt.xlabel('Training Steps')
            plt.ylabel('Loss')
            plt.legend()
            plt.yscale('log')
            plt.tight_layout()
            plt.savefig('training_curves.png')
            print("Successfully saved training curves plot")
        else:
            print("No valid training curves data found for plotting")
            
    except Exception as e:
        print(f"Error creating training curves plot: {str(e)}")

def plot_group_mi():
    """Plot group mutual information trends."""
    plt.figure()
    
    for run, label in labels.items():
        _, results = load_run_data(run)
        if results and 'training_log' in results:
            mi_losses = [log.get('mi_loss', float('nan')) for log in results['training_log']]
            steps = range(len(mi_losses))
            plt.plot(steps, mi_losses, label=label)
    
    plt.title('Group Mutual Information Over Training')
    plt.xlabel('Training Steps')
    plt.ylabel('MI Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('group_mi.png')

def plot_feature_metrics():
    """Plot feature isolation metrics."""
    metrics = {
        'sparsity_loss': [],
        'l2_loss': [],
        'mi_loss': []
    }
    runs = []
    
    for run, label in labels.items():
        _, results = load_run_data(run)
        if results and 'training_log' in results and results['training_log']:  # Check if log exists and not empty
            try:
                log = results['training_log'][-1]  # Get final metrics
                for metric in metrics:
                    metrics[metric].append(log.get(metric, float('nan')))
                runs.append(label)
            except (IndexError, KeyError) as e:
                print(f"Warning: Could not process metrics for {label}: {str(e)}")
    
    if runs:
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            for ax, (metric, values) in zip(axes, metrics.items()):
                ax.bar(runs, values)
                ax.set_title(f'Final {metric.replace("_", " ").title()}')
                ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            plt.savefig('feature_metrics.png')
            print(f"Successfully saved feature metrics plot with {len(runs)} runs")
        except Exception as e:
            print(f"Error creating feature metrics plot: {str(e)}")
    else:
        print("No valid run data found for feature metrics plot")

def main():
    """Generate all plots."""
    plot_unlearning_scores()
    plot_training_curves()
    plot_group_mi()
    plot_feature_metrics()

if __name__ == "__main__":
    main()
