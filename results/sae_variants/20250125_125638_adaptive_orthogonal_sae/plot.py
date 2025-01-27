import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Define labels for runs we want to plot
labels = {
    "run_4": "Initial Loss Tracking",
    "run_5": "Loss Calculation Fix",
    "run_6": "Loss Computation Debug",
    "run_7": "Training Loop Validation",
    "run_8": "Buffer Fix Attempt",
    "run_9": "Step Counter Fix",
    "run_10": "Gradient Flow Debug"
}

def load_training_log(run_dir):
    """Load training log for a run."""
    log_path = os.path.join(run_dir, "training_log.jsonl")
    if not os.path.exists(log_path):
        return None
        
    metrics = defaultdict(list)
    steps = []
    
    with open(log_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                steps.append(data['step'])
                for k, v in data.items():
                    if k != 'step':
                        metrics[k].append(float(v))
            except:
                continue
                
    return steps, metrics

def make_plots():
    """Generate plots comparing runs."""
    metrics_to_plot = [
        'total_loss',
        'l2_loss', 
        'sparsity_loss',
        'ortho_loss',
        'active_features',
        'unique_features_in_pairs',
        'max_pair_correlation',
        'mean_pair_correlation'
    ]
    
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        
        for run_dir, label in labels.items():
            result = load_training_log(run_dir)
            if result is not None:
                steps, metrics = result
                if metric in metrics:
                    plt.plot(steps, metrics[metric], label=label)
        
        plt.xlabel('Training Steps')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'{metric.replace("_", " ").title()} vs Training Steps')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'{metric}_comparison.png')
        plt.close()

def plot_final_metrics():
    """Plot final metrics across runs."""
    final_metrics = defaultdict(list)
    run_names = []
    
    for run_dir in labels.keys():
        info_path = os.path.join(run_dir, "final_info.json")
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                data = json.load(f)
                if 'final_metrics' in data:
                    metrics = data['final_metrics']
                    for k, v in metrics.items():
                        if v is not None:
                            final_metrics[k].append(float(v))
                        else:
                            final_metrics[k].append(0.0)
                    run_names.append(labels[run_dir])
    
    if run_names:
        plt.figure(figsize=(12, 6))
        x = np.arange(len(run_names))
        width = 0.15
        
        for i, (metric, values) in enumerate(final_metrics.items()):
            plt.bar(x + i*width, values, width, label=metric)
        
        plt.xlabel('Runs')
        plt.ylabel('Final Metric Values')
        plt.title('Final Metrics Comparison Across Runs')
        plt.xticks(x + width * (len(final_metrics)-1)/2, run_names, rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('final_metrics_comparison.png')
        plt.close()

if __name__ == "__main__":
    make_plots()
    plot_final_metrics()
