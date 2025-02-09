import matplotlib.pyplot as plt
import numpy as np

def plot_competition_dynamics(fast_avg, slow_avg, save_path):
    """Plot fast and slow moving averages of feature competition"""
    plt.figure(figsize=(10, 6))
    plt.plot(fast_avg, label='Fast MA', alpha=0.7)
    plt.plot(slow_avg, label='Slow MA', alpha=0.7)
    plt.xlabel('Training Steps')
    plt.ylabel('Competition Strength')
    plt.title('Dual-Timescale Feature Competition Dynamics')
    plt.legend()
    plt.savefig(save_path)
    plt.close()
