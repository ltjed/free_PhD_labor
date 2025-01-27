import matplotlib.pyplot as plt

def plot_cosine_schedule(total_steps):
    """Plot cosine annealing schedule for lambda and k"""
    steps = torch.arange(total_steps)
    lambda_vals = 0.5 * (1 + torch.cos(torch.pi * steps / total_steps))
    
    plt.figure(figsize=(10, 5))
    plt.plot(steps, lambda_vals, label='Lambda(t)')
    plt.title('Cosine Annealing Schedule')
    plt.xlabel('Training Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('cosine_schedule.png')
    plt.close()
