import matplotlib.pyplot as plt
import numpy as np

def plot_feature_separation(separation_history, thresholds, save_path):
    """Plot feature separation metrics over training."""
    plt.figure(figsize=(10, 6))
    
    # Plot average feature separation
    plt.plot(separation_history['steps'], separation_history['avg_separation'], 
             label='Avg Feature Separation', color='blue')
    
    # Plot threshold progression
    plt.plot(separation_history['steps'], thresholds,
             label='Orthogonality Threshold', color='red', linestyle='--')
    
    plt.xlabel('Training Step')
    plt.ylabel('Metric Value')
    plt.title('Feature Separation Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
