import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Style configuration
plt.style.use('seaborn-v0_8-darkgrid')  # Using a valid style name
sns.set_theme()  # This sets up seaborn defaults
plt.rcParams.update({
    'figure.figsize': [12, 8],
    'font.size': 12,
    'axes.grid': True,
    'grid.alpha': 0.3
})

# Define runs to analyze and their labels
labels = {
    'run_1': 'Initial Hierarchical SAE',
    'run_2': 'Simplified Architecture',
    'run_3': 'Enhanced Initialization',
    'run_4': 'Debug Logging',
    'run_5': 'Batch Norm + Gradient Scaling'
}

def load_run_data(run_dir):
    """Load training data and final info for a run."""
    try:
        with open(os.path.join(run_dir, 'final_info.json'), 'r') as f:
            final_info = json.load(f)
        
        results_path = os.path.join(run_dir, 'all_results.npy')
        if os.path.exists(results_path):
            with open(results_path, 'rb') as f:
                results = np.load(f, allow_pickle=True).item()
            return final_info, results
        return final_info, None
    except:
        return None, None

def plot_training_metrics():
    """Plot training metrics across runs."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    for run_name, label in labels.items():
        final_info, results = load_run_data(run_name)
        if results and 'training_log' in results:
            steps = range(len(results['training_log']))
            losses = [log.get('loss', float('nan')) for log in results['training_log']]
            l2_losses = [log.get('l2_loss', float('nan')) for log in results['training_log']]
            sparsity_losses = [log.get('sparsity_loss', float('nan')) for log in results['training_log']]
            mse_losses = [log.get('mse_loss', float('nan')) for log in results['training_log']]
            
            ax1.plot(steps, losses, label=label)
            ax2.plot(steps, l2_losses, label=label)
            ax3.plot(steps, sparsity_losses, label=label)
            ax4.plot(steps, mse_losses, label=label)
    
    ax1.set_title('Total Loss')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.set_title('L2 Loss')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('L2 Loss')
    ax2.legend()
    
    ax3.set_title('Sparsity Loss')
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('L1 Penalty')
    ax3.legend()
    
    ax4.set_title('MSE Loss')
    ax4.set_xlabel('Training Steps')
    ax4.set_ylabel('MSE')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

def plot_final_metrics():
    """Plot final metrics comparison across runs."""
    metrics = []
    runs = []
    
    for run_name, label in labels.items():
        final_info, _ = load_run_data(run_name)
        if final_info:
            # Replace None values with NaN
            metrics.append([
                final_info.get('training_steps', 0),
                float('nan') if final_info.get('final_loss') is None else final_info.get('final_loss'),
                final_info.get('learning_rate', 0),
                final_info.get('sparsity_penalty', 0)
            ])
            runs.append(label)
    
    if metrics:
        metrics = np.array(metrics, dtype=float)  # Ensure float dtype for NaN support
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        ax1.bar(runs, metrics[:, 0])
        ax1.set_title('Training Steps Completed')
        ax1.tick_params(axis='x', rotation=45)
        
        ax2.bar(runs, metrics[:, 1])
        ax2.set_title('Final Loss')
        ax2.tick_params(axis='x', rotation=45)
        
        ax3.bar(runs, metrics[:, 2])
        ax3.set_title('Learning Rate')
        ax3.tick_params(axis='x', rotation=45)
        
        ax4.bar(runs, metrics[:, 3])
        ax4.set_title('Sparsity Penalty')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('final_metrics.png')
        plt.close()

def plot_core_eval_metrics():
    """Plot core evaluation metrics across runs."""
    metrics = {}
    
    for run_name, label in labels.items():
        eval_dir = os.path.join(run_name, 'eval_results')
        if os.path.exists(eval_dir):
            for file in os.listdir(eval_dir):
                if file.endswith('_core.json'):
                    with open(os.path.join(eval_dir, file), 'r') as f:
                        data = json.load(f)
                        if 'eval_result_metrics' in data:
                            metrics[label] = data['eval_result_metrics']
    
    if metrics:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        runs = list(metrics.keys())
        
        # Plot reconstruction quality
        recon_scores = [m['reconstruction_quality']['explained_variance'] for m in metrics.values()]
        ax1.bar(runs, recon_scores)
        ax1.set_title('Reconstruction Quality\n(Explained Variance)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot sparsity
        sparsity_scores = [m['sparsity']['l0'] for m in metrics.values()]
        ax2.bar(runs, sparsity_scores)
        ax2.set_title('Sparsity (L0)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot KL divergence
        kl_scores = [m['model_behavior_preservation']['kl_div_score'] for m in metrics.values()]
        ax3.bar(runs, kl_scores)
        ax3.set_title('Model Behavior Preservation\n(KL Divergence)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot CE loss
        ce_scores = [m['model_performance_preservation']['ce_loss_score'] for m in metrics.values()]
        ax4.bar(runs, ce_scores)
        ax4.set_title('Model Performance\n(Cross Entropy Loss)')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('core_eval_metrics.png')
        plt.close()

if __name__ == '__main__':
    # Create plots
    plot_training_metrics()
    plot_final_metrics()
    plot_core_eval_metrics()
