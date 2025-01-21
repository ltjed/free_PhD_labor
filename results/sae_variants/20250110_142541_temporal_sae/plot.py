import matplotlib.pyplot as plt
import numpy as np
import torch
import json
import os

def plot_temporal_consistency(run_dir):
    """Plot temporal consistency metrics."""
    with open(os.path.join(run_dir, "final_info.json"), "r") as f:
        results = json.load(f)
    
    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(results.get('training_log', []))
    plt.title('Training Losses Over Time')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend(['Reconstruction', 'L1 Sparsity', 'Temporal Consistency'])
    plt.savefig(os.path.join(run_dir, 'temporal_losses.png'))
    plt.close()

def plot_feature_transitions(sae, run_dir):
    """Plot feature transition patterns."""
    W_temporal = sae.W_temporal.detach().cpu().numpy()
    
    plt.figure(figsize=(10, 10))
    plt.imshow(W_temporal, cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Feature Transition Matrix')
    plt.xlabel('Next Feature')
    plt.ylabel('Current Feature')
    plt.savefig(os.path.join(run_dir, 'feature_transitions.png'))
    plt.close()
