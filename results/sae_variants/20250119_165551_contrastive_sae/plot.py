import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from pathlib import Path

# Style settings
plt.style.use('default')
plt.rcParams.update({
    'figure.figsize': [12, 8],
    'font.size': 12,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.facecolor': '#f0f0f0',
    'figure.facecolor': 'white'
})
colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']

# Dictionary mapping run directories to their display labels
labels = {
    'run_1': 'Initial Contrastive (w=0.1)',
    'run_2': 'Enhanced Contrastive (w=0.5)', 
    'run_3': 'Gradient Stabilization',
    'run_4': 'Extended Training',
    'run_5': 'Feature Diversity (w=2.0)'
}

def load_run_data(run_dir):
    """Load results from a run directory."""
    info_path = os.path.join(run_dir, 'final_info.json')
    if not os.path.exists(info_path):
        return None
    
    with open(info_path, 'r') as f:
        return json.load(f)

def plot_unlearning_scores():
    """Plot unlearning scores across runs."""
    scores = []
    names = []
    
    for run_dir, label in labels.items():
        data = load_run_data(run_dir)
        if data and 'eval_result_metrics' in data:
            score = data['eval_result_metrics'].get('unlearning', {}).get('unlearning_score', 0)
            scores.append(score)
            names.append(label)
    
    if not scores:
        return
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(names, scores, color=colors[:len(names)])
    plt.title('Unlearning Scores Across Experimental Runs')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Unlearning Score')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('unlearning_scores.png')
    plt.close()

def plot_training_metrics():
    """Plot training steps and final loss across runs."""
    steps = []
    losses = []
    names = []
    
    for run_dir, label in labels.items():
        data = load_run_data(run_dir)
        if data:
            steps.append(data.get('training_steps', 0))
            # Handle None values for losses
            loss = data.get('final_loss')
            losses.append(0.0 if loss is None else float(loss))
            names.append(label)
    
    if not steps:
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Set x-axis ticks first
    x = np.arange(len(names))
    
    # Training steps plot
    bars1 = ax1.bar(x, steps, color=colors[:len(names)])
    ax1.set_title('Training Steps per Run')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.set_ylabel('Number of Steps')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    # Final loss plot
    bars2 = ax2.bar(x, losses, color=colors[:len(names)])
    ax2.set_title('Final Loss per Run')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.set_ylabel('Loss Value')
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

def main():
    """Generate all plots."""
    plot_unlearning_scores()
    plot_training_metrics()

if __name__ == '__main__':
    main()
