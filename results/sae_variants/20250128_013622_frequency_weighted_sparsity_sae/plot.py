import json
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

# Style settings
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']

# Define labels for each run
labels = {
    'run_0': 'Baseline (JumpReLU)',
    'run_1': 'Freq-Weight (α=0.1)',
    'run_2': 'Freq-Weight (α=0.2)', 
    'run_3': 'Freq-Weight (α=0.5)',
    'run_4': 'Freq-Weight Extended'
}

def load_results(run_dir):
    """Load results from a run directory."""
    try:
        with open(os.path.join(run_dir, 'final_info.json'), 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"No results found in {run_dir}")
        return None

def extract_metrics(data):
    """Extract key metrics from run data."""
    if not data:
        return None
    
    layer_data = data.get('training results for layer 19', {})
    core_metrics = layer_data.get('core evaluation results', {}).get('metrics', {})
    
    return {
        'kl_div': core_metrics.get('model_behavior_preservation', {}).get('kl_div_score'),
        'explained_var': core_metrics.get('reconstruction_quality', {}).get('explained_variance'),
        'l0_sparsity': core_metrics.get('sparsity', {}).get('l0'),
        'final_loss': layer_data.get('final_info', {}).get('final_loss')
    }

def create_metric_comparison(metric_name, metric_values, ylabel):
    """Create a bar plot comparing a metric across runs."""
    plt.figure(figsize=(10, 6))
    
    # Filter out None values
    filtered_values = {k: v for k, v in metric_values.items() if v is not None}
    runs = list(filtered_values.keys())
    values = list(filtered_values.values())
    
    if not values:  # Skip if no valid values
        print(f"No valid values for {metric_name}, skipping plot")
        plt.close()
        return
        
    bars = plt.bar(runs, values, color=colors[:len(runs)])
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.title(f'{metric_name} Comparison Across Runs')
    plt.xlabel('Run')
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    
    # Replace x-axis labels with descriptive names
    ax = plt.gca()
    ax.set_xticklabels([labels.get(run, run) for run in runs])
    
    plt.tight_layout()
    plt.savefig(f'{metric_name.lower().replace(" ", "_")}.png')
    plt.close()

def main():
    # Collect results from all runs
    all_metrics = {}
    for run_dir in labels.keys():
        results = load_results(run_dir)
        metrics = extract_metrics(results)
        if metrics:
            all_metrics[run_dir] = metrics
    
    # Prepare data for plotting
    metric_data = {
        'KL Divergence': {run: metrics['kl_div'] for run, metrics in all_metrics.items()},
        'Explained Variance': {run: metrics['explained_var'] for run, metrics in all_metrics.items()},
        'L0 Sparsity': {run: metrics['l0_sparsity'] for run, metrics in all_metrics.items()},
        'Final Loss': {run: metrics['final_loss'] for run, metrics in all_metrics.items()}
    }
    
    # Create plots
    for metric_name, values in metric_data.items():
        create_metric_comparison(
            metric_name,
            values,
            'Score' if metric_name != 'Final Loss' else 'Loss'
        )
    
    # Create combined plot
    plt.figure(figsize=(15, 10))
    
    metrics_to_plot = ['KL Divergence', 'Explained Variance']
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(1, 2, i+1)
        values = metric_data[metric]
        # Filter out None values
        filtered_values = {k: v for k, v in values.items() if v is not None}
        runs = list(filtered_values.keys())
        plt.bar(runs, list(filtered_values.values()), color=colors[:len(runs)])
        plt.title(metric)
        plt.xticks(rotation=45)
        ax = plt.gca()
        ax.set_xticklabels([labels.get(run, run) for run in runs])
    
    plt.tight_layout()
    plt.savefig('combined_metrics.png')
    plt.close()

if __name__ == '__main__':
    main()
