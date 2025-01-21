import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path

def load_results(results_dir):
    """Load results from the evaluation directories."""
    results = {}
    for eval_type in ['absorption', 'autointerp', 'core', 'scr', 'tpp', 'sparse_probing', 'unlearning']:
        eval_path = Path(results_dir) / eval_type
        if eval_path.exists():
            results[eval_type] = {}
            for result_file in eval_path.glob('*.json'):
                with open(result_file) as f:
                    results[eval_type][result_file.stem] = json.load(f)
    return results

def plot_metrics(results, output_dir='plots'):
    """Generate plots for various metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot reconstruction loss
    if 'core' in results:
        losses = results['core'].get('reconstruction_loss', {})
        if losses:
            plt.figure(figsize=(10, 6))
            plt.plot(losses.values())
            plt.title('Reconstruction Loss Over Time')
            plt.xlabel('Training Steps')
            plt.ylabel('Loss')
            plt.savefig(os.path.join(output_dir, 'reconstruction_loss.png'))
            plt.close()
    
    # Plot sparsity
    if 'core' in results:
        sparsity = results['core'].get('sparsity_metrics', {})
        if sparsity:
            plt.figure(figsize=(10, 6))
            plt.plot(sparsity.values())
            plt.title('Feature Sparsity Over Time')
            plt.xlabel('Training Steps')
            plt.ylabel('Sparsity')
            plt.savefig(os.path.join(output_dir, 'sparsity.png'))
            plt.close()

def main():
    results_dir = 'eval_results'
    results = load_results(results_dir)
    plot_metrics(results)

if __name__ == '__main__':
    main()
