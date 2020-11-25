import matplotlib.pyplot as plt
import numpy as np
import random
from collections import namedtuple

from torch._C import dtype


def plot_winsratio(wins, title, window_size=50):
    """
    Plots total win ratio + moving average for the last 50 matches
    """
    # parameters
    cumulative_wins = np.cumsum(wins, dtype=float)
    indx = np.array([i + 1 for i in range(len(wins))], dtype=float)

    # plot
    plt.figure(2)
    plt.clf()
    plt.title(f"Training {title}")
    plt.xlabel("Episodes")
    plt.ylabel("WR")
    plt.plot(cumulative_wins / indx, label="Total WR")
    # Take 100 episode averages and plot them too
    if len(wins) >= window_size:
        means = cumulative_wins
        means[window_size:] = means[window_size:] - means[:-window_size]
        plt.plot(means[window_size - 1 :] / window_size, label="Running average WR")

    plt.legend()
    plt.savefig("imgs/train_ai.png")


def rgb2grayscale(rgb: np.ndarray):
    """
    transform rgb image to gray scale
    """
    # transform to rgb
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    grayscale = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return grayscale


Transition = namedtuple("Transition", ("ob", "action", "next_ob", "rew", "done"))


class ReplayMemory(object):
    """
    Replay memory used for experience replay.
    It stores transitions.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args) -> None:
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> np.ndarray:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)