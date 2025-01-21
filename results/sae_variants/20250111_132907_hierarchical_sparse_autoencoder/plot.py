import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
import networkx as nx
import json
import os
from pathlib import Path

# Dictionary mapping run numbers to their descriptions
labels = {
    "run_1": "Basic Hierarchical Implementation",
    "run_2": "Hyperparameter Tuning",
    "run_3": "Path Attribution Analysis",
    "run_4": "Adaptive Sparsity",
    "run_5": "Feature Specialization"
}

def load_run_results(run_dir: str) -> Dict:
    """Load results from a run directory."""
    info_path = os.path.join(run_dir, "final_info.json")
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            return json.load(f)
    return None

def plot_training_progression():
    """Plot metrics across all runs to show progression of improvements."""
    metrics = {}
    
    # Collect metrics from each run
    for run_id in labels.keys():
        results = load_run_results(run_id)
        if results:
            metrics[run_id] = results
    
    # Create subplots for different metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training Progression Across Runs', fontsize=16)
    
    # Plot loss
    ax = axes[0, 0]
    losses = [m.get('final_loss', 0) for m in metrics.values()]
    ax.plot(range(1, len(losses)+1), losses, 'o-')
    ax.set_xlabel('Run Number')
    ax.set_ylabel('Final Loss')
    ax.set_title('Loss Progression')
    
    # Plot sparsity
    ax = axes[0, 1]
    sparsity = [m.get('sparsity_penalty', 0) for m in metrics.values()]
    ax.plot(range(1, len(sparsity)+1), sparsity, 'o-')
    ax.set_xlabel('Run Number')
    ax.set_ylabel('Sparsity Penalty')
    ax.set_title('Sparsity Adjustment')
    
    # Add run labels
    ax = axes[1, 0]
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, [1]*len(labels))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(list(labels.values()))
    ax.set_title('Run Descriptions')
    ax.invert_yaxis()
    
    # Clear unused subplot
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('training_progression.png')
    plt.close()

def plot_feature_paths(
    sae_model,
    activations: torch.Tensor,
    top_k: int = 5,
    threshold: float = 0.1,
    save_path: str = None,
    show_specialization: bool = True
) -> None:
    """Plot the attribution paths through the hierarchical levels.
    
    Args:
        sae_model: The trained hierarchical SAE model
        activations: Input activations to analyze
        top_k: Number of top features to highlight per level
        threshold: Minimum attribution value to include in visualization
        save_path: Optional path to save the plot
    """
    # Get features and attributions
    with torch.no_grad():
        _, features = sae_model(activations, output_features=True)
    
    # Create graph
    G = nx.DiGraph()
    pos = {}
    
    # Add nodes for each level
    level_heights = np.linspace(0, 1, len(features) + 1)
    for level_idx, feature_tensor in enumerate(features):
        # Get top k active features
        mean_activity = feature_tensor.abs().mean(0)
        top_features = mean_activity.topk(min(top_k, len(mean_activity))).indices
        
        # Add nodes
        width = len(top_features)
        for i, feat_idx in enumerate(top_features):
            node_id = f"L{level_idx}_{feat_idx}"
            G.add_node(node_id, 
                      level=level_idx,
                      activity=mean_activity[feat_idx].item())
            pos[node_id] = (i - width/2, level_heights[level_idx])
    
    # Add edges between levels
    for level_idx in range(len(features)-1):
        source_nodes = [n for n in G.nodes if f"L{level_idx}_" in n]
        target_nodes = [n for n in G.nodes if f"L{level_idx+1}_" in n]
        
        # Compute attributions between levels
        for s in source_nodes:
            s_idx = int(s.split('_')[1])
            for t in target_nodes:
                t_idx = int(t.split('_')[1])
                attribution = compute_attribution(
                    sae_model, level_idx, s_idx, t_idx
                )
                if attribution > threshold:
                    G.add_edge(s, t, weight=attribution)
    
    # Plot
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos,
            node_color=[G.nodes[n]['activity'] for n in G.nodes],
            edge_color=[G.edges[e]['weight'] for e in G.edges],
            width=[G.edges[e]['weight'] * 5 for e in G.edges],
            node_size=1000,
            with_labels=True,
            cmap=plt.cm.viridis)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def compute_attribution(
    sae_model,
    level_idx: int,
    source_idx: int,
    target_idx: int
) -> float:
    """Compute attribution between features in adjacent levels."""
    # Get weights from level transformation
    weights = sae_model.level_transforms[level_idx].weight
    return abs(weights[target_idx, source_idx].item())

def plot_level_activations(
    features: List[torch.Tensor],
    save_path: str = None,
    show_utilization: bool = True
) -> None:
    """Plot activation patterns for each hierarchical level."""
    n_levels = len(features)
    fig, axes = plt.subplots(n_levels, 1, figsize=(12, 4*n_levels))
    
    if n_levels == 1:
        axes = [axes]
    
    for i, (ax, feat) in enumerate(zip(axes, features)):
        sns.heatmap(
            feat.detach().cpu().numpy(),
            ax=ax,
            cmap='viridis',
            cbar_kws={'label': 'Activation'},
        )
        ax.set_title(f'Level {i+1} Activations')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Sample Index')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()
if __name__ == "__main__":
    # Generate comprehensive plots for the writeup
    print("Generating training progression plot...")
    plot_training_progression()
    
    print("Plots generated successfully!")
    print("Output files:")
    print("- training_progression.png: Shows metrics across all runs")
    print("- feature_paths.png: Shows hierarchical feature relationships")
    print("- level_activations.png: Shows activation patterns per level")
