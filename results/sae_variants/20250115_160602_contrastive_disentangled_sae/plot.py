import matplotlib.pyplot as plt
import numpy as np

def plot_loss_components(losses, save_path):
    """Plot training losses over time"""
    plt.figure(figsize=(10, 6))
    for key in losses[0].keys():
        values = [d[key] for d in losses]
        plt.plot(values, label=key)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()
