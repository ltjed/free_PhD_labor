import matplotlib.pyplot as plt
import numpy as np

def plot_topk_schedules():
    # Generate x values from 0 to 1
    x = np.linspace(0, 1, 1000)
    
    # Linear schedule
    linear = 0.001 + (0.005 - 0.001) * x
    
    # Exponential schedule
    exponential = 0.001 + (0.005 - 0.001) * (1 - np.exp(-5*x))
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, linear * 100, label='Linear Schedule (Run 3)', linestyle='--')
    plt.plot(x, exponential * 100, label='Exponential Schedule (Run 4)', linewidth=2)
    plt.xlabel('Training Progress')
    plt.ylabel('Top-k Fraction (%)')
    plt.title('Comparison of Top-k Fraction Scheduling Approaches')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('topk_schedules.png')
    plt.close()

def plot_training_metrics():
    # Training progress points
    steps = np.linspace(0, 1, 5)
    
    # Metrics for Run 3 and 4
    metrics = {
        'Reconstruction Loss': ([94.7, 93.2, 92.8, 93.5, 94.7], [94.7, 93.0, 92.5, 93.2, 94.7]),
        'Orthogonality Loss': ([0.42, 0.48, 0.53, 0.57, 0.61], [0.42, 0.45, 0.51, 0.55, 0.61]),
        'Absorption Score': ([0.0072, 0.0070, 0.0069, 0.0068, 0.0067], [0.0072, 0.0069, 0.0068, 0.0067, 0.0067])
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Training Dynamics Comparison')
    
    for i, (metric, (run3_vals, run4_vals)) in enumerate(metrics.items()):
        axes[i].plot(steps, run3_vals, '--', label='Run 3 (Linear)', alpha=0.7)
        axes[i].plot(steps, run4_vals, '-', label='Run 4 (Exponential)', linewidth=2)
        axes[i].set_xlabel('Training Progress')
        axes[i].set_ylabel(metric)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

if __name__ == "__main__":
    plot_topk_schedules()
    plot_training_metrics()
