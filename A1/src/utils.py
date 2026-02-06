import matplotlib.pyplot as plt
import numpy as np
import os

def moving_average(x, window=100):
    return np.convolve(x, np.ones(window)/window, mode='valid')

def plot_loss_curve(flat_losses, save_dir):
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    smoothed = moving_average(flat_losses, window=1000)

    plt.figure()
    plt.plot(smoothed)
    plt.xlabel("Training Step")
    plt.ylabel("Smoothed Loss")
    plt.title("Smoothed Training Loss")
    plt.grid(True)

    # Full save path
    save_path = os.path.join(save_dir, filename)

    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Loss curve saved to {save_path}")