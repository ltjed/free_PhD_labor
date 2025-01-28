import matplotlib.pyplot as plt
import numpy as np

def plot_frequency_consistency_results():
    # Plot 1: Training Progress
    baseline_loss = 7932.06
    fc_loss = 4293.39
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Loss Comparison
    losses = [baseline_loss, fc_loss]
    labels = ['Baseline', 'Frequency-Consistency']
    ax1.bar(labels, losses)
    ax1.set_title('Final Loss Comparison')
    ax1.set_ylabel('Loss')
    for i, v in enumerate(losses):
        ax1.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    
    # Feature Metrics
    metrics = {
        'Explained Variance': [0.824, 0.926],
        'KL Divergence Score': [0.989, 0.996],
        'CE Loss Score': [0.990, 0.997]
    }
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax2.bar(x - width/2, [metrics[m][0] for m in metrics], width, label='Baseline')
    ax2.bar(x + width/2, [metrics[m][1] for m in metrics], width, label='Frequency-Consistency')
    
    ax2.set_ylabel('Score')
    ax2.set_title('Model Performance Metrics')
    ax2.set_xticks(x)
    ax2.set_xticklabels(list(metrics.keys()), rotation=45)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('frequency_consistency_results.png')
    plt.close()

if __name__ == "__main__":
    plot_frequency_consistency_results()
