# write your code hereimport matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_adaptation_params(results_path: str, out_dir: str = "plots"):
    """Plot evolution of global adaptation parameters."""
    data = np.load(results_path, allow_pickle=True).item()
    params = {
        'beta': [],
        'theta': [],
        'temperature': [],
        'steps': []
    }
    
    for step_log in data['training_log']:
        params['beta'].append(step_log.get('beta', 1.0))
        params['theta'].append(step_log.get('theta', 0.0))
        params['temperature'].append(step_log.get('temperature', 0.5))
        params['steps'].append(step_log.get('step', 0))
    
    plt.figure(figsize=(12, 6))
    plt.plot(params['steps'], params['beta'], label='Beta')
    plt.plot(params['steps'], params['theta'], label='Theta')
    plt.plot(params['steps'], params['temperature'], label='Temperature')
    plt.xlabel('Training Steps')
    plt.ylabel('Parameter Value')
    plt.title('Global Adaptation Parameters Over Time')
    plt.legend()
    plt.savefig(f"{out_dir}/adaptation_params.png")
    plt.close()

def plot_variance_scaling(activations: np.ndarray, out_dir: str = "plots"):
    """Visualize feature variance distributions."""
    plt.figure(figsize=(10, 6))
    sns.histplot(activations.var(axis=0), kde=True)
    plt.xlabel('Feature Variance')
    plt.ylabel('Count')
    plt.title('Pre/Post Variance Scaling Distribution')
    plt.savefig(f"{out_dir}/variance_distribution.png")
    plt.close()
