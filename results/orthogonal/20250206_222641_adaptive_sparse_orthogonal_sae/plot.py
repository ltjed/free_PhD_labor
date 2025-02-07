import matplotlib.pyplot as plt
import numpy as np
import json
import os

def plot_feature_separation(log_dir):
    """Plot feature separation metrics over training progress"""
    results = []
    for filename in os.listdir(log_dir):
        if filename.startswith("feature_separation_") and filename.endswith(".json"):
            with open(os.path.join(log_dir, filename)) as f:
                results.append(json.load(f))
    
    if not results:
        return
        
    progress = [r["progress"] for r in results]
    ortho_loss = [r["orthogonality_loss"] for r in results]
    avg_overlap = [r["average_overlap"] for r in results]
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(progress, ortho_loss)
    plt.xlabel("Training Progress")
    plt.ylabel("Orthogonality Loss")
    plt.title("Orthogonality Loss vs Progress")
    
    plt.subplot(1, 2, 2)
    plt.plot(progress, avg_overlap)
    plt.xlabel("Training Progress") 
    plt.ylabel("Average Feature Overlap")
    plt.title("Feature Overlap vs Progress")
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "feature_separation.png"))
    plt.close()
