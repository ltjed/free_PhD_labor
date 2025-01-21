import matplotlib.pyplot as plt
import numpy as np

def plot_nces_distribution(nces_scores, save_path):
    plt.figure(figsize=(10, 6))
    plt.hist(nces_scores, bins=50, density=True)
    plt.xlabel('Normalized Causal Effect Score (NCES)')
    plt.ylabel('Density')
    plt.title('Distribution of NCES Across Features')
    plt.savefig(save_path)
    plt.close()

def plot_intervention_effects(original, intervened, reconstructed, save_path):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(original.T, aspect='auto')
    plt.title('Original Activations')
    
    plt.subplot(132)
    plt.imshow(intervened.T, aspect='auto')
    plt.title('Intervened Activations')
    
    plt.subplot(133)
    plt.imshow(reconstructed.T, aspect='auto')
    plt.title('Reconstructed Activations')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
