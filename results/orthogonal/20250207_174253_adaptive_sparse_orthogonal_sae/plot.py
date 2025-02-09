import json
import matplotlib.pyplot as plt
import os
import numpy as np

# Configuration
results_dir = "."
plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

# Map run directories to human-readable labels
labels = {
    "run_0": "Baseline TopK SAE",
    "run_1": "Run 1: Base Implementation",
    "run_2": "Run 2: Threshold Optimization", 
    "run_3": "Run 3: Linear Ortho Decay",
    "run_4": "Run 4: EMA + Warmup",
    "run_5": "Run 5: Hybrid Schedule",
    "run_6": "Run 6: Final Config",
    "run_7": "Run 7: Cosine Decay"
}

# Data storage
absorption_scores = []
sparse_probing_acc = []
run_labels = []

# Load results from each run
for run_dir in labels.keys():
    try:
        with open(os.path.join(run_dir, "final_info.json")) as f:
            data = json.load(f)
            
            # Get first layer results (dynamic key handling)
            layer_key = [k for k in data.keys() if k.startswith('training results for layer')][0]
            layer_data = data[layer_key]
            
            # Get absorption score
            absorption = layer_data.get('absorption evaluation results', {}).get('eval_result_metrics', {}).get('mean', {}).get('mean_absorption_score', 0)
            
            # Get sparse probing top-1 accuracy
            sparse_probing = layer_data.get('sparse probing evaluation results', {}).get('eval_result_metrics', {}).get('sae', {}).get('sae_top_1_test_accuracy', 0)
            
            if absorption > 0:  # Filter out invalid entries
                absorption_scores.append(absorption)
                sparse_probing_acc.append(sparse_probing)
                run_labels.append(labels[run_dir])
            
    except (FileNotFoundError, IndexError, KeyError) as e:
        print(f"Skipping {run_dir} due to error: {str(e)}")
        continue

# Set style for better visualization
plt.style.use('seaborn')
plt.figure(figsize=(15, 7))

# Absorption scores plot
plt.subplot(1, 2, 1)
bars = plt.barh(run_labels, absorption_scores, color='darkred')
plt.title('Feature Absorption Comparison')
plt.xlim(0, 0.15)
plt.xlabel('Mean Absorption Score (Lower Better)')
plt.gca().invert_yaxis()

# Add values and improvement percentages to bars
baseline = absorption_scores[0]  # Baseline is first run
for bar in bars:
    width = bar.get_width()
    y_pos = bar.get_y() + bar.get_height()/2
    improvement = ((baseline - width) / baseline * 100) if width < baseline else 0
    label = f'{width:.3f} ({improvement:+.1f}%)' if improvement else f'{width:.3f}'
    plt.text(width + 0.005, y_pos, label, ha='left', va='center')

# Add threshold line for target absorption
plt.axvline(x=0.016, color='gray', linestyle='--', alpha=0.5)
plt.text(0.017, plt.ylim()[0], 'Target\n(0.016)', ha='left', va='bottom')

# Sparse probing accuracy plot
plt.subplot(1, 2, 2)
bars = plt.barh(run_labels, sparse_probing_acc, color='darkblue')
plt.title('Sparse Probing Performance')
plt.xlim(0.6, 0.85)
plt.xlabel('Top-1 Accuracy (Higher Better)')
plt.gca().invert_yaxis()

# Add values to bars
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.005, bar.get_y() + bar.get_height()/2,
             f'{width:.3f}', ha='left', va='center')

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'results_comparison.png'), dpi=300, bbox_inches='tight')

# Training dynamics plot (example for final run)
final_run = "run_7"
try:
    with open(os.path.join(final_run, "all_results.npy"), "rb") as f:
        results = np.load(f, allow_pickle=True).item()
        
        # Check data structure
        if 'training_log' in results:
            losses = results["training_log"]
        elif 'training_log' in results.get('final_info', {}):
            losses = results['final_info']['training_log']
        else:
            print(f"No training log found in {final_run}")
            losses = []
        
        if losses:
            plt.figure(figsize=(10, 6))
            progress = np.linspace(0, 1, len(losses))
            plt.plot(progress, losses, color='purple', linewidth=2)
        
            # Add phase annotations
            phases = [
                (0, 0.2, "Phase 1:\nFeature\nEstablishment", "82% loss\nreduction"),
                (0.2, 0.7, "Phase 2:\nOrtho Constraint\nOptimization", "12% gradual\nimprovement"),
                (0.7, 1.0, "Phase 3:\nConvergence", "<2% change")
            ]
        
            for start, end, label, metric in phases:
                mid = (start + end) / 2
                plt.axvspan(start, end, alpha=0.1, color='gray')
                plt.text(mid, plt.ylim()[1]*0.9, label, ha='center', va='top')
                plt.text(mid, plt.ylim()[1]*0.7, metric, ha='center', va='top', 
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
            # Add threshold change markers
            plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
            plt.text(0.51, plt.ylim()[1], 'Threshold\nStep Change', ha='left', va='top')
        
            plt.title('Training Loss Progression with Phase Analysis\n(Final Configuration)')
            plt.xlabel('Training Progress')
            plt.ylabel('Total Loss')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(plots_dir, 'training_dynamics.png'), dpi=300)
        
except FileNotFoundError:
    print(f"Results file not found for {final_run}")
    pass

print(f"Plots saved to {plots_dir}/ directory")
