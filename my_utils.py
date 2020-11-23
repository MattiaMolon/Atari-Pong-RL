import matplotlib.pyplot as plt
import numpy as np


def plot_winsratio(wins, title, window_size=10):
    plt.figure(2)
    plt.clf()
    plt.title(f"Training {title}")
    plt.xlabel("Episode")
    plt.ylabel("Wins")
    plt.plot(wins)
    # Take 10 episode averages and plot them too
    if len(wins) >= 10:
        means = np.cumsum(wins, dtype="float")
        means[window_size:] = means[window_size:] - means[:-window_size]
        plt.plot(means[window_size - 1 :] / window_size)

    plt.savefig("imgs/train_ai.png")


def rgb2grayscale(rgb: np.ndarray):

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    grayscale = 0.2989 * r + 0.5870 * g + 0.1140 * b
    grayscale = np.expand_dims(grayscale, 0)

    # TODO: flip image if id != 1
    return grayscale
