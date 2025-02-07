import matplotlib.pyplot as plt
import numpy as np

def plot_feature_progression(logs):
    """Plot training metrics over time."""
    steps = range(len(logs))
    metrics = {
        'loss': [],
        'l2_loss': [],
        'ortho_loss': [],
        'current_threshold': [],
        'lambda_ortho': []
    }
    
    for log in logs:
        for key in metrics:
            if key in log:
                metrics[key].append(log[key])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot losses
    ax1.plot(steps, metrics['loss'], label='Total Loss')
    ax1.plot(steps, metrics['l2_loss'], label='L2 Loss')
    ax1.plot(steps, metrics['ortho_loss'], label='Orthogonality Loss')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Training Losses')
    
    # Plot progression parameters
    ax2.plot(steps, metrics['current_threshold'], label='Similarity Threshold')
    ax2.plot(steps, metrics['lambda_ortho'], label='Orthogonality Weight')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Parameter Value')
    ax2.legend()
    ax2.set_title('Progressive Parameters')
    
    plt.tight_layout()
    plt.savefig('training_progression.png')
    plt.close()
