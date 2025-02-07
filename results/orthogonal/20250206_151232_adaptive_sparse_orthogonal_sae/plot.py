import matplotlib.pyplot as plt
import json
import numpy as np

def plot_training_progression(run_dir):
    # Load results
    with open(f"{run_dir}/final_info.json", 'r') as f:
        results = json.load(f)
    
    # Extract metrics
    training_results = results["training results for layer 12"]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot metrics
    metrics = training_results.get("metrics", {})
    plt.plot(metrics.get("progress", []), label="Training Progress")
    plt.plot(metrics.get("ortho_loss", []), label="Orthogonality Loss")
    
    plt.xlabel("Training Steps")
    plt.ylabel("Value")
    plt.title("Training Progression")
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig(f"{run_dir}/training_progression.png")
    plt.close()

if __name__ == "__main__":
    import sys
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "run_1"
    plot_training_progression(run_dir)
