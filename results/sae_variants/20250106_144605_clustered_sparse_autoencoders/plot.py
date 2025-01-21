import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score

def plot_clustering_results(activations_by_layer, cluster_labels, save_path):
    """Plot clustering visualization and metrics"""
    # Compute 2D PCA projection of layer representations
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    
    # Stack all layer activations
    all_activations = np.vstack(activations_by_layer)
    projected = pca.fit_transform(all_activations)
    
    # Plot clusters
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(projected[:, 0], projected[:, 1], 
                         c=cluster_labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title('Layer Clusters (PCA projection)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    
    # Add silhouette score
    score = silhouette_score(all_activations, cluster_labels)
    plt.text(0.02, 0.98, f'Silhouette Score: {score:.3f}',
             transform=plt.gca().transAxes)
    
    plt.savefig(save_path)
    plt.close()
    
    return score
