import matplotlib.pyplot as plt
import numpy as np
import json
import os
from typing import Dict, List, Optional

# Define labels for runs to include in plots
labels = {
}

def load_run_results(run_dir: str) -> Optional[Dict]:
    """Load results from a run directory."""
    results_path = os.path.join(run_dir, "all_results.npy")
    if not os.path.exists(results_path):
        return None
    with open(results_path, 'rb') as f:
        return np.load(f, allow_pickle=True).item()

def plot_comparative_analysis(base_dir: str):
    """Generate comparative plots across runs."""
    plt.figure(figsize=(15, 10))
    
    # Track metrics across runs
    final_losses = []
    sparsity_levels = []
    run_names = []
    
    for run_id, run_label in labels.items():
        results = load_run_results(os.path.join(base_dir, run_id))
        if results is None:
            continue
            
        training_log = results.get('training_log', [])
        if not training_log:
            continue
            
        # Extract final metrics
        final_losses.append(training_log[-1].get('losses', {}).get('loss', 0))
        sparsity_levels.append(training_log[-1].get('losses', {}).get('sparsity_loss', 0))
        run_names.append(run_label)
    
    # Plot final loss comparison
    plt.subplot(2, 2, 1)
    plt.bar(range(len(run_names)), final_losses)
    plt.xticks(range(len(run_names)), run_names, rotation=45, ha='right')
    plt.ylabel('Final Loss')
    plt.title('Loss Comparison Across Runs')
    
    # Plot sparsity comparison
    plt.subplot(2, 2, 2)
    plt.bar(range(len(run_names)), sparsity_levels)
    plt.xticks(range(len(run_names)), run_names, rotation=45, ha='right')
    plt.ylabel('Sparsity Level')
    plt.title('Sparsity Comparison Across Runs')
    
    plt.tight_layout()
    plt.savefig('comparative_analysis.png')
    plt.close()

def plot_position_losses(out_dir: str):
    """Plot position-wise loss patterns and weight evolution from training results."""
    
    # Load results
    results_path = os.path.join(out_dir, "all_results.npy")
    if not os.path.exists(results_path):
        print(f"No results file found at {results_path}")
        return
        
    with open(results_path, 'rb') as f:
        results = np.load(f, allow_pickle=True).item()
    
    training_log = results.get('training_log', [])
    if not training_log:
        print("No training log found in results")
        return
        
    # Extract position-wise losses and weights
    pos_losses = []
    pos_weights = []
    for entry in training_log:
        if isinstance(entry, dict):
            if 'losses' in entry and 'avg_pos_loss' in entry['losses']:
                pos_losses.append(entry['losses']['avg_pos_loss'])
            if 'pos_weights' in entry:
                pos_weights.append(entry['pos_weights'])
    
    if not pos_losses:
        print("No position-wise losses found in training log")
        return
        
    pos_losses = np.array(pos_losses)
    pos_weights = np.array(pos_weights) if pos_weights else None
    
    # Create visualization
    plt.figure(figsize=(15, 5))
    
    # Plot average loss evolution per position
    plt.subplot(1, 3, 1)
    plt.imshow(pos_losses.T, aspect='auto', cmap='viridis')
    plt.colorbar(label='Loss')
    plt.xlabel('Training Step')
    plt.ylabel('Sequence Position')
    plt.title('Position-wise Loss Evolution')
    
    # Plot final position-wise loss distribution
    plt.subplot(1, 3, 2)
    final_losses = pos_losses[-1]
    plt.plot(final_losses, '-o')
    plt.xlabel('Sequence Position')
    plt.ylabel('Loss')
    plt.title('Final Position-wise Loss Distribution')
    
    # Plot position weight evolution if available
    if pos_weights is not None:
        plt.subplot(1, 3, 3)
        plt.imshow(pos_weights.T, aspect='auto', cmap='RdYlBu')
        plt.colorbar(label='Weight')
        plt.xlabel('Training Step')
        plt.ylabel('Sequence Position')
        plt.title('Position Weight Evolution')
    
    # Create new figure for sparsity analysis
    plt.figure(figsize=(15, 5))
    
    # Extract sparsity patterns if available
    feature_acts = []
    for entry in training_log:
        if isinstance(entry, dict) and 'f' in entry:
            feature_acts.append(entry['f'])
    
    if feature_acts:
        feature_acts = np.array(feature_acts)
        
        # Plot temporal sparsity evolution
        plt.subplot(1, 3, 1)
        sparsity = (feature_acts == 0).mean(axis=2)  # Average across features
        plt.imshow(sparsity.T, aspect='auto', cmap='viridis')
        plt.colorbar(label='Sparsity Rate')
        plt.xlabel('Training Step')
        plt.ylabel('Sequence Position')
        plt.title('Position-wise Sparsity Evolution')
        
        # Plot final sparsity distribution
        plt.subplot(1, 3, 2)
        final_sparsity = sparsity[-1]
        plt.plot(final_sparsity, '-o')
        plt.xlabel('Sequence Position')
        plt.ylabel('Sparsity Rate')
        plt.title('Final Position-wise Sparsity')
        
        # Plot feature activity heatmap
        plt.subplot(1, 3, 3)
        final_acts = feature_acts[-1].mean(axis=0)  # Average across batch
        plt.imshow(final_acts, aspect='auto', cmap='RdYlBu')
        plt.colorbar(label='Average Activation')
        plt.xlabel('Feature Index')
        plt.ylabel('Sequence Position')
        plt.title('Feature Activity by Position')
    
    # Perform temporal feature clustering if sufficient data available
    if feature_acts is not None and len(feature_acts) > 0:
        # Create new figure for feature clustering analysis
        plt.figure(figsize=(15, 10))
        
        # Get final feature activities
        final_acts = feature_acts[-1]  # Shape: (batch, pos, features)
        
        # Compute temporal activation profiles
        temporal_profiles = final_acts.mean(axis=0)  # Average across batch
        
        # Perform hierarchical clustering
        from scipy.cluster import hierarchy
        from scipy.spatial.distance import pdist
        
        # Compute distance matrix based on temporal activation patterns
        distances = pdist(temporal_profiles.T, metric='correlation')
        linkage = hierarchy.linkage(distances, method='ward')
        
        # Plot dendrogram
        plt.subplot(2, 1, 1)
        hierarchy.dendrogram(linkage, leaf_rotation=90)
        plt.title('Feature Clustering by Temporal Activation Patterns')
        plt.xlabel('Feature Index')
        plt.ylabel('Distance')
        
        # Plot sorted feature heatmap
        plt.subplot(2, 1, 2)
        ordered_idx = hierarchy.leaves_list(linkage)
        sorted_profiles = temporal_profiles[:, ordered_idx]
        plt.imshow(sorted_profiles, aspect='auto', cmap='RdYlBu')
        plt.colorbar(label='Average Activation')
        plt.xlabel('Clustered Feature Index')
        plt.ylabel('Sequence Position')
        plt.title('Temporally Clustered Feature Activities')
        
        # Analyze activation patterns within clusters
        plt.figure(figsize=(15, 10))
        
        # Get cluster assignments
        n_clusters = 5  # Number of top-level clusters to analyze
        cluster_labels = hierarchy.fcluster(linkage, n_clusters, criterion='maxclust')
        
        # Compute cluster statistics
        for cluster_id in range(1, n_clusters + 1):
            cluster_mask = cluster_labels == cluster_id
            cluster_features = temporal_profiles[:, cluster_mask]
            
            # Plot average activation pattern for cluster
            plt.subplot(n_clusters, 2, (cluster_id-1)*2 + 1)
            mean_pattern = cluster_features.mean(axis=1)
            std_pattern = cluster_features.std(axis=1)
            positions = np.arange(len(mean_pattern))
            plt.plot(positions, mean_pattern, 'b-', label='Mean')
            plt.fill_between(positions, 
                           mean_pattern - std_pattern,
                           mean_pattern + std_pattern,
                           alpha=0.3, color='b')
            plt.title(f'Cluster {cluster_id} Pattern')
            plt.xlabel('Position')
            plt.ylabel('Activation')
            
            # Plot activation consistency within cluster
            plt.subplot(n_clusters, 2, (cluster_id-1)*2 + 2)
            consistency = np.corrcoef(cluster_features.T)
            plt.imshow(consistency, cmap='RdYlBu')
            plt.colorbar(label='Correlation')
            plt.title(f'Cluster {cluster_id} Consistency')
            
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'cluster_patterns.png'))
        plt.close()
        
        # Save original clustering analysis
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'temporal_clustering.png'))
        plt.close()
    
    # Save sparsity analysis plot
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'sparsity_analysis.png'))
    plt.close()
    
    # Save original position analysis plot
    plt.figure(figsize=(15, 5))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'position_analysis.png'))
    plt.close()
