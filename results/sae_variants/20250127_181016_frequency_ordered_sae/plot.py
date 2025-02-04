import torch
import matplotlib.pyplot as plt
import numpy as np
import json
import os

# Dictionary mapping run numbers to descriptive labels
labels = {
    "2": "Frequency Ordering",
    "3": "Adaptive Penalty",
    "4": "Enhanced Resampling", 
    "5": "Adaptive L1",
    "6": "Gradient Clipping",
    "7": "Layer Normalization",
    "8": "Skip Connections",
    "9": "Self-Attention"
}

def load_run_results(run_number):
    """Load results from a specific run directory."""
    try:
        with open(f"run_{run_number}/final_info.json", 'r') as f:
            data = json.load(f)
            return data[f"training results for layer 19"]
    except FileNotFoundError:
        return None

def plot_training_losses():
    """Plot final training losses across runs."""
    losses = []
    runs = []
    
    for run in sorted(labels.keys(), key=int):
        results = load_run_results(run)
        if results and 'final_info' in results:
            losses.append(results['final_info']['final_loss'])
            runs.append(labels[run])
    
    plt.figure(figsize=(12, 6))
    plt.bar(runs, losses)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Final Training Loss')
    plt.title('Training Loss Comparison Across Model Iterations')
    plt.tight_layout()
    plt.savefig('training_loss_comparison.png')
    plt.close()

def plot_reconstruction_metrics():
    """Plot reconstruction quality metrics across runs."""
    kl_scores = []
    cosine_sims = []
    runs = []
    
    for run in sorted(labels.keys(), key=int):
        results = load_run_results(run)
        if results and 'core evaluation results' in results:
            metrics = results['core evaluation results']['metrics']
            kl_scores.append(metrics['model_behavior_preservation']['kl_div_score'])
            cosine_sims.append(metrics['reconstruction_quality']['cossim'])
            runs.append(labels[run])
    
    plt.figure(figsize=(12, 6))
    width = 0.35
    x = np.arange(len(runs))
    
    plt.bar(x - width/2, kl_scores, width, label='KL Divergence Score')
    plt.bar(x + width/2, cosine_sims, width, label='Cosine Similarity')
    
    plt.xlabel('Model Version')
    plt.ylabel('Score')
    plt.title('Reconstruction Quality Metrics Across Model Iterations')
    plt.xticks(x, runs, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('reconstruction_metrics.png')
    plt.close()

def plot_absorption_metrics():
    """Plot absorption evaluation metrics across runs."""
    absorption_scores = []
    split_features = []
    runs = []
    
    for run in sorted(labels.keys(), key=int):
        results = load_run_results(run)
        if results and 'absorption evaluation results' in results:
            metrics = results['absorption evaluation results']['eval_result_metrics']['mean']
            absorption_scores.append(metrics['mean_absorption_score'])
            split_features.append(metrics['mean_num_split_features'])
            runs.append(labels[run])
    
    plt.figure(figsize=(12, 6))
    width = 0.35
    x = np.arange(len(runs))
    
    plt.bar(x - width/2, absorption_scores, width, label='Mean Absorption Score')
    plt.bar(x + width/2, split_features, width, label='Mean Split Features')
    
    plt.xlabel('Model Version')
    plt.ylabel('Score')
    plt.title('Absorption Metrics Across Model Iterations')
    plt.xticks(x, runs, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('absorption_metrics.png')
    plt.close()

def plot_scr_thresholds():
    """Plot SCR metrics across different thresholds."""
    thresholds = [2, 5, 10, 20]
    plt.figure(figsize=(12, 6))
    
    for run in sorted(labels.keys(), key=int):
        results = load_run_results(run)
        if results and 'scr and tpp evaluations results' in results:
            metrics = results['scr and tpp evaluations results']['eval_result_metrics']['scr_metrics']
            scores = [metrics[f'scr_dir1_threshold_{t}'] for t in thresholds]
            plt.plot(thresholds, scores, marker='o', label=labels[run])
    
    plt.xlabel('Feature Threshold')
    plt.ylabel('SCR Score')
    plt.title('SCR Performance Across Feature Thresholds')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('scr_threshold_comparison.png')
    plt.close()

def plot_feature_frequencies(sae, save_path):
    """Plot feature activation frequencies."""
    if sae.total_batches == 0:
        print("No frequency data available")
        return
        
    freqs = sae.activation_counts.cpu().numpy() / sae.total_batches
    
    plt.figure(figsize=(10, 5))
    plt.plot(freqs, '-')
    plt.xlabel('Feature Index')
    plt.ylabel('Activation Frequency')
    plt.title('Feature Activation Frequencies')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_frequency_ordering(sae, save_path):
    """Plot features ordered by frequency."""
    if sae.total_batches == 0:
        print("No frequency data available")
        return
        
    freqs = sae.activation_counts.cpu().numpy() / sae.total_batches
    sorted_freqs = np.sort(freqs)[::-1]
    
    plt.figure(figsize=(10, 5))
    plt.plot(sorted_freqs, '-')
    plt.xlabel('Rank')
    plt.ylabel('Activation Frequency')
    plt.title('Features Ordered by Frequency')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    
    # Generate all comparison plots
    plot_training_losses()
    plot_reconstruction_metrics()
    plot_absorption_metrics()
    plot_scr_thresholds()
