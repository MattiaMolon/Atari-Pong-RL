import matplotlib.pyplot as plt
import numpy as np
import random
from collections import namedtuple

from torch._C import dtype


def plot_winsratio(wins, title, wsize_mean=200, wsize_means_mean=100):
    """
    Plots moving average WR for the last 100 matches
    """
    if len(wins) >= wsize_mean:

        # Take 100 episode averages
        means = np.cumsum(wins, dtype=float)
        means[wsize_mean:] = means[wsize_mean:] - means[:-wsize_mean]
        means = means[wsize_mean - 1 :] / wsize_mean
        idxs = [i + wsize_mean - 1 for i in range(len(means))]
        plt.plot(idxs, means, label=f"Running {wsize_mean} average WR")

        # Take 20 episode averages of the 100 running average
        if len(means) >= wsize_means_mean:
            means_mean = np.cumsum(means)
            means_mean[wsize_means_mean:] = (
                means_mean[wsize_means_mean:] - means_mean[:-wsize_means_mean]
            )
            means_mean = means_mean[wsize_means_mean - 1 :] / wsize_means_mean
            idxs_mean = [
                i + wsize_mean + wsize_means_mean - 2 for i in range(len(means_mean))
            ]
            plt.plot(
                idxs_mean,
                means_mean,
                label=f"Running {wsize_mean} average WR mean",
            )

        plt.legend()
        plt.savefig("imgs/train_ai.png")
        plt.close()


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

    def __init__(self, memory_capacity: int, buffer_capacity: int) -> None:
        self.memory_capacity = memory_capacity
        self.buffer_capacity = buffer_capacity
        self.memory = []
        self.buffer = []
        self.memory_position = 0

    def push_to_memory(self, *args) -> None:
        """Saves a transition."""
        if len(self.memory) < self.memory_capacity:
            self.memory.append(None)
        self.memory[self.memory_position] = Transition(*args)
        self.memory_position = (self.memory_position + 1) % self.memory_capacity

    def push_to_buffer(self, *args) -> None:
        """Saves a transition."""
        self.buffer.append(Transition(*args))
        if len(self.buffer) > self.buffer_capacity:
            raise Exception("Error: capacity of the buffer exceded")

    def sample(self, batch_size: int) -> np.ndarray:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)