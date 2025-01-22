import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def plot_feature_groups(features, group_assignments, save_path):
    """Plot hierarchical feature group structure"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot features colored by group
    for level, assignments in enumerate(group_assignments):
        unique_groups = np.unique(assignments)
        for group in unique_groups:
            mask = assignments == group
            ax.scatter(features[mask, 0], features[mask, 1], 
                      alpha=0.5, label=f'Level {level} Group {group}')
    
    ax.set_xlabel('Feature Dimension 1')
    ax.set_ylabel('Feature Dimension 2') 
    ax.set_title('Hierarchical Feature Groups')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
