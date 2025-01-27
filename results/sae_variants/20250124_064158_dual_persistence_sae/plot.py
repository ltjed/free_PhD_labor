import matplotlib.pyplot as plt
import numpy as np

def plot_dual_streaks(active_streaks, inactive_streaks, save_path=None):
    """Plot dual streak distributions with log scale."""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(np.log1p(active_streaks.cpu().numpy()), bins=50)
    plt.title("Active Streak Distribution (log scale)")
    plt.xlabel("log(streak length + 1)")
    plt.ylabel("Count")
    
    plt.subplot(1, 2, 2)
    plt.hist(np.log1p(inactive_streaks.cpu().numpy()), bins=50)
    plt.title("Inactive Streak Distribution (log scale)")
    plt.xlabel("log(streak length + 1)")
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_streak_correlation(active_streaks, inactive_streaks, save_path=None):
    """Plot correlation between active and inactive streaks."""
    plt.figure(figsize=(8, 6))
    plt.hexbin(active_streaks.cpu().numpy(), 
              inactive_streaks.cpu().numpy(), 
              gridsize=50, 
              cmap='viridis',
              bins='log')
    plt.colorbar()
    plt.title("Active-Inactive Streak Correlation")
    plt.xlabel("Active Streak Length")
    plt.ylabel("Inactive Streak Length")
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
