import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_condition_numbers(condition_numbers, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(condition_numbers)
    plt.xlabel('Training Steps')
    plt.ylabel('Condition Number')
    plt.title('Feature Subspace Condition Numbers')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
