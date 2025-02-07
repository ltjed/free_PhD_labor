import matplotlib.pyplot as plt
import numpy as np
import json
import os

def load_results(run_dir):
    """Load results from a run directory."""
    with open(os.path.join(run_dir, "final_info.json"), "r") as f:
        return json.load(f)

def plot_training_metrics(runs):
    """Plot training metrics across different runs."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    for run_name, results in runs.items():
        training_log = results.get("training_log", [])
        steps = range(len(training_log))
        
        # Extract metrics
        losses = [log.get("losses", {}) for log in training_log]
        orth_losses = [l.get("orth_loss", 0) for l in losses]
        active_feats = [l.get("active_features", 0) for l in losses]
        lambda2s = [l.get("lambda2", 0) for l in losses]
        total_losses = [l.get("loss", 0) for l in losses]
        
        # Plot metrics
        ax1.plot(steps, orth_losses, label=f"{run_name} Orth Loss")
        ax2.plot(steps, active_feats, label=f"{run_name} Active Features")
        ax3.plot(steps, lambda2s, label=f"{run_name} Lambda2")
        ax4.plot(steps, total_losses, label=f"{run_name} Total Loss")
    
    ax1.set_title("Orthogonality Loss")
    ax2.set_title("Active Features")
    ax3.set_title("Lambda2 Progression")
    ax4.set_title("Total Loss")
    
    for ax in (ax1, ax2, ax3, ax4):
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.close()

if __name__ == "__main__":
    # Load results from different runs
    runs = {}
    for run_dir in ["run_1", "run_2", "run_3", "run_4", "run_5"]:
        if os.path.exists(run_dir):
            runs[run_dir] = load_results(run_dir)
    
    if runs:
        plot_training_metrics(runs)
