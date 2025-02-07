import matplotlib.pyplot as plt
import numpy as np
import json
import os

def plot_feature_separation(log_dir):
    """Plot feature separation metrics over training progress"""
    with open(os.path.join(log_dir, "feature_separation.json"), "r") as f:
        data = json.load(f)
    
    progress = data["progress"]
    ortho_loss = data["orthogonality_loss"]
    avg_cosine = data["average_cosine_similarity"]
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(progress, ortho_loss)
    plt.xlabel("Training Progress")
    plt.ylabel("Orthogonality Loss")
    plt.title("Orthogonality Loss vs Progress")
    
    plt.subplot(1, 2, 2)
    plt.plot(progress, avg_cosine)
    plt.xlabel("Training Progress") 
    plt.ylabel("Avg Cosine Similarity")
    plt.title("Feature Similarity vs Progress")
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "feature_separation.png"))
    plt.close()
