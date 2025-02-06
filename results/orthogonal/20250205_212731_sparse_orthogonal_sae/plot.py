import matplotlib.pyplot as plt

def plot_absorption(run_data):
    """
    Simple function to visually compare mean_absorption_score across runs.
    run_data: dict of {run_number: mean_absorption_score}
    """
    runs = sorted(run_data.keys())
    scores = [run_data[r] for r in runs]

    plt.figure(figsize=(7,4))
    plt.plot(runs, scores, marker='o', linewidth=2)
    plt.title("Mean Absorption Score Across Runs")
    plt.xlabel("Run Number")
    plt.ylabel("Mean Absorption Score")
    plt.ylim(bottom=0)
    plt.grid(True)
    plt.show()
