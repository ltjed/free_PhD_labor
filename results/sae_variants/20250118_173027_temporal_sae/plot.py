import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path

# Dictionary mapping run names to display labels
labels = {
    "Run 0": "Baseline SAE",
    "Run 1": "Basic MTSAE",
    "Run 2": "Optimized MTSAE",
    "Run 3": "Expanded Context MTSAE"
}

def load_results(notes_file="notes.txt"):
    """Load results from notes.txt"""
    results = {}
    current_run = None
    
    with open(notes_file, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        if line.startswith('## Run'):
            current_run = line.split(':')[0].strip()
            results[current_run] = {}
        elif current_run and 'Results:' in line:
            # Parse the results dictionary string
            try:
                results_str = line.split('Results:')[1].strip()
                results_dict = eval(results_str)
                results[current_run]['metrics'] = results_dict
            except:
                print(f"Could not parse results for {current_run}")

    return results

def plot_accuracy_comparison(results):
    """Plot accuracy comparison across runs"""
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(labels))
    width = 0.35
    
    llm_accuracies = []
    sae_accuracies = []
    
    for run in labels.keys():
        if run in results and 'metrics' in results[run]:
            metrics = results[run]['metrics']
            if 'eval_result_metrics' in metrics:
                llm_accuracies.append(metrics['eval_result_metrics']['llm']['llm_test_accuracy'])
                sae_accuracies.append(metrics['eval_result_metrics']['sae']['sae_test_accuracy'])
    
    if len(llm_accuracies) > 0:
        plt.bar(x - width/2, llm_accuracies, width, label='LLM Accuracy')
        plt.bar(x + width/2, sae_accuracies, width, label='SAE Accuracy')
    else:
        print("Warning: No accuracy metrics found in results")
        return
    
    plt.xlabel('Model Version')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison Across Model Versions')
    plt.xticks(x, [labels[run] for run in labels.keys()])
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png')
    plt.close()

def plot_loss_curves(results):
    """Plot training loss curves"""
    plt.figure(figsize=(10, 6))
    
    has_data = False
    for run in labels.keys():
        if run in results and 'metrics' in results[run]:
            metrics = results[run]['metrics']
            if 'training_steps' in metrics and metrics['training_steps'] > 0:
                steps = range(metrics['training_steps'])
                if 'final_loss' in metrics and metrics['final_loss'] is not None:
                    plt.plot(steps, [metrics['final_loss']] * len(steps), 
                            label=labels[run])
                    has_data = True
    
    if not has_data:
        print("Warning: No loss data found in results")
        return
    
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Curves')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('loss_curves.png')
    plt.close()

def plot_sparsity_comparison(results):
    """Plot sparsity penalty comparison"""
    plt.figure(figsize=(8, 6))
    
    sparsity_values = []
    for run in labels.keys():
        if run in results and 'metrics' in results[run]:
            metrics = results[run]['metrics']
            if 'sparsity_penalty' in metrics and metrics['sparsity_penalty'] is not None:
                sparsity_values.append(metrics['sparsity_penalty'])
    
    if len(sparsity_values) > 0:
        plt.bar(range(len(sparsity_values)), sparsity_values)
    else:
        print("Warning: No sparsity penalty data found in results")
        return
    plt.xticks(range(len(sparsity_values)), [labels[run] for run in labels.keys()])
    plt.xlabel('Model Version')
    plt.ylabel('Sparsity Penalty')
    plt.title('Sparsity Penalty Comparison')
    
    plt.tight_layout()
    plt.savefig('sparsity_comparison.png')
    plt.close()

def main():
    # Create output directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Load results
    results = load_results()
    
    # Generate plots
    plot_accuracy_comparison(results)
    plot_loss_curves(results)
    plot_sparsity_comparison(results)
    
    print("Plots have been generated in the 'plots' directory")

if __name__ == "__main__":
    main()
