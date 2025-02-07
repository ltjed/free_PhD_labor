import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("tab10")

# Configuration
OUTPUT_DIR = "analysis_plots"
METRIC_NAMES = [
    "explained_variance",
    "l0",
    "kl_div_score",
    "mean_absorption_score"
]
LABELS = {
    "run_0": "Baseline TopK (k=320)",
    "run_1": "Progressive v1 (λ=1e-3)",
    "run_2": "Progressive v2 (λ=5e-3)", 
    "run_3": "Threshold Adjusted",
    "run_4": "Gradient Clamped",
    "run_5": "Exponential Decay",
    "run_final": "Final Config (k=320 fixed)"
}

def load_run_data(run_dir):
    """Load evaluation results from a run directory"""
    results = {}
    run_path = Path(run_dir)
    
    # Load core metrics
    core_file = run_path / "core_eval_results.npy"
    if core_file.exists():
        core_data = np.load(core_file, allow_pickle=True).item()
        results.update({
            "explained_variance": core_data['metrics']['reconstruction_quality']['explained_variance'],
            "l0": core_data['metrics']['sparsity']['l0'],
            "kl_div_score": core_data['metrics']['model_behavior_preservation']['kl_div_score']
        })
    
    # Load absorption metrics
    absorption_file = run_path / "absorption_eval_results.npy" 
    if absorption_file.exists():
        absorption_data = np.load(absorption_file, allow_pickle=True).item()
        results["mean_absorption_score"] = absorption_data['eval_result_metrics']['mean']['mean_absorption_score']
        
    return results

def plot_metrics(all_results):
    """Generate comparative analysis plots"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Reconstruction Quality Comparison
    plt.figure(figsize=(10,6))
    for run_name, data in all_results.items():
        if 'explained_variance' in data:
            plt.plot(data['explained_variance'], label=LABELS[run_name], alpha=0.8)
    plt.title("Feature Reconstruction Quality (Explained Variance)")
    plt.xlabel("Training Steps (×100)")
    plt.ylabel("Explained Variance")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/reconstruction_quality.png", dpi=300)
    plt.close()

    # 2. Sparsity Dynamics
    plt.figure(figsize=(10,6))
    for run_name, data in all_results.items():
        if 'l0' in data:
            plt.plot(data['l0'], label=LABELS[run_name], linestyle='--' if 'Baseline' in LABELS[run_name] else '-')
    plt.title("Active Features During Training (L0 Sparsity)")
    plt.xlabel("Training Steps (×100)")
    plt.ylabel("Number of Active Features")
    plt.axhline(y=320, color='gray', linestyle=':', label='Target Sparsity')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/sparsity_dynamics.png", dpi=300)
    plt.close()

    # 3. Behavioral Preservation
    kl_scores = {LABELS[k]: v['kl_div_score'] for k,v in all_results.items() if 'kl_div_score' in v}
    plt.figure(figsize=(10,6))
    sns.barplot(x=list(kl_scores.values()), y=list(kl_scores.keys()), palette="viridis")
    plt.title("Model Behavior Preservation (KL Divergence)")
    plt.xlabel("KL Divergence Score (Lower = Better)")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/kl_divergence.png", dpi=300)
    plt.close()

    # 4. Feature Separation Analysis
    absorption_scores = {LABELS[k]: v['mean_absorption_score'] for k,v in all_results.items() if 'mean_absorption_score' in v}
    plt.figure(figsize=(10,6))
    sns.barplot(x=list(absorption_scores.values()), y=list(absorption_scores.keys()), palette="rocket")
    plt.title("Feature Separation Performance")
    plt.xlabel("Mean Absorption Score (Lower = Better)")
    plt.axvline(x=0.01, color='red', linestyle='--', label='Target Threshold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feature_separation.png", dpi=300)
    plt.close()

def plot_feature_similarity(cos_sim, step):
    plt.figure(figsize=(10,8))
    sns.heatmap(cos_sim.cpu().numpy(), vmin=0, vmax=0.5, cmap='viridis')
    plt.title(f"Feature Cosine Similarity @ Step {step}")
    plt.savefig(f"cos_sim_step_{step}.png")
    plt.close()

if __name__ == "__main__":
    # Load data from all runs
    all_results = {}
    for run_dir in LABELS.keys():
        if Path(run_dir).exists():
            all_results[run_dir] = load_run_data(run_dir)
    
    # Generate plots
    plot_metrics(all_results)
    print(f"Generated plots saved to {OUTPUT_DIR}/")
