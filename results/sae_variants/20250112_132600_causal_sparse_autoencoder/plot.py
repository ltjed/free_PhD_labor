import matplotlib.pyplot as plt
import numpy as np

def plot_nces_scores(nces_scores, feature_indices, title="NCES Scores by Feature"):
    plt.figure(figsize=(10, 6))
    plt.bar(feature_indices, nces_scores)
    plt.xlabel("Feature Index")
    plt.ylabel("NCES Score")
    plt.title(title)
    plt.tight_layout()
    plt.savefig("nces_scores.png")
    plt.close()

def plot_intervention_effects(original, intervened, reconstructed, n_samples=5):
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 3*n_samples))
    for i in range(n_samples):
        axes[i,0].imshow(original[i].reshape(-1, 1), aspect='auto')
        axes[i,0].set_title('Original' if i == 0 else '')
        axes[i,1].imshow(intervened[i].reshape(-1, 1), aspect='auto')
        axes[i,1].set_title('Intervened' if i == 0 else '')
        axes[i,2].imshow(reconstructed[i].reshape(-1, 1), aspect='auto')
        axes[i,2].set_title('Reconstructed' if i == 0 else '')
    plt.tight_layout()
    plt.savefig("intervention_effects.png")
    plt.close()
