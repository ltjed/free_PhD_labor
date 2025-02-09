import matplotlib.pyplot as plt
import numpy as np

def plot_rank_correlations(correlations, window_size, out_dir):
    plt.figure(figsize=(10, 6))
    plt.hist(correlations.flatten(), bins=50)
    plt.title(f'Distribution of Feature Rank Correlations (Window={window_size})')
    plt.xlabel('Rank Correlation')
    plt.ylabel('Count')
    plt.savefig(f'{out_dir}/rank_correlations.png')
    plt.close()
