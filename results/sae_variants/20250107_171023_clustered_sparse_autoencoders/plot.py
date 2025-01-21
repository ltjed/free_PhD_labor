import numpy as np
import torch
from sklearn.cluster import KMeans
from typing import List, Dict, Tuple

def cluster_layers(activations_dict: Dict[int, torch.Tensor], n_clusters: int) -> Dict[int, List[int]]:
    """
    Cluster layers based on their activation patterns.
    Returns dict mapping cluster_id -> list of layer indices
    """
    # Stack activations from all layers
    layer_features = []
    layer_indices = []
    
    for layer_idx, acts in activations_dict.items():
        # Compute summary statistics for clustering
        mean = acts.mean(dim=0).cpu().numpy()
        std = acts.std(dim=0).cpu().numpy()
        layer_features.append(np.concatenate([mean, std]))
        layer_indices.append(layer_idx)
        
    features = np.stack(layer_features)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(features)
    
    # Group layers by cluster
    cluster_to_layers = {}
    for cluster_idx in range(n_clusters):
        mask = clusters == cluster_idx
        cluster_to_layers[cluster_idx] = [layer_indices[i] for i in range(len(layer_indices)) if mask[i]]
        
    return cluster_to_layers

def plot_cluster_metrics(cluster_metrics: Dict[int, Dict[str, float]], save_path: str):
    """Plot metrics for each cluster"""
    import matplotlib.pyplot as plt
    
    metrics = ['reconstruction_loss', 'sparsity']
    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 4))
    
    for i, metric in enumerate(metrics):
        values = [m[metric] for m in cluster_metrics.values()]
        axes[i].bar(range(len(values)), values)
        axes[i].set_title(metric)
        axes[i].set_xlabel('Cluster')
        axes[i].set_ylabel('Value')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
