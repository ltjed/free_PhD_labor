import matplotlib.pyplot as plt
import numpy as np

def plot_temporal_metrics(log_file):
    """Plot temporal stability metrics over training."""
    data = np.load(log_file)
    steps = range(len(data['temporal_consistency']))
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, data['temporal_consistency'], label='Temporal Consistency')
    plt.plot(steps, data['reconstruction_loss'], label='Reconstruction Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Metrics Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('temporal_metrics.png')
    plt.close()
