import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple


def plot_winsratio(wins, title, window_size=10):
    """
    Plots total win ratio + moving average for the last 10 matches
    """
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


def rgb2grayscale(rgb: np.ndarray, side):
    """
    transform rgb image to gray scale + it flips it if the game
    is played in the opposite side in respect to the training
    """
    # use correct side
    if side != 1:
        rgb = np.fliplr(rgb)

    # transform to rgb
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    grayscale = 0.2989 * r + 0.5870 * g + 0.1140 * b
    grayscale = np.expand_dims(grayscale, 0)

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