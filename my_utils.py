import matplotlib.pyplot as plt
import numpy as np
import random
from collections import namedtuple


def plot_winsratio(
    wins: list,
    title: str,
    start_idx: int = 0,
    wsize_mean: int = 100,
    wsize_means_mean: int = 1000,
    opponent_update_idxs=None,
):
    """Winrate plotting function, plots both a the WR over the last wsize_mean episodes and
    a WR mean over the last wsize_means_mean wsize_mean episodes

    Args:
        wins (list): Wins vector. Contains 0 or 1 for each loss or victory
        title (str): Title to use in the plot
        start_idx (int, optional): Start for the x labels. Defaults to 0.
        wsize_mean (int, optional): Window size to compute the Winrate. Defaults to 100.
        wsize_means_mean (int, optional): Window size to compute the mean over the winrates. Defaults to 1000.
        opponent_updates_idxs (list, optional): List of indexes where the update of the opponent state dict has happened in self play. Default None.
    """
    if len(wins) >= wsize_mean:

        # Take 100 episode averages
        means = np.cumsum(wins, dtype=float)
        means[wsize_mean:] = means[wsize_mean:] - means[:-wsize_mean]
        means = means[wsize_mean - 1 :] / wsize_mean
        idxs = [i + start_idx + wsize_mean - 1 for i in range(len(means))]
        plt.plot(idxs, means, label=f"Running {wsize_mean} average WR")

        # Take 20 episode averages of the 100 running average
        if len(means) >= wsize_means_mean:
            means_mean = np.cumsum(means)
            means_mean[wsize_means_mean:] = (
                means_mean[wsize_means_mean:] - means_mean[:-wsize_means_mean]
            )
            means_mean = means_mean[wsize_means_mean - 1 :] / wsize_means_mean
            idxs_mean = [
                i + start_idx + wsize_mean + wsize_means_mean - 2
                for i in range(len(means_mean))
            ]
            plt.plot(
                idxs_mean,
                means_mean,
                label=f"Running {wsize_mean} average WR mean",
            )

        # add vertical lines for opponent update during self play
        if opponent_update_idxs != None:
            for x in opponent_update_idxs:
                if x >= wsize_mean:
                    plt.axvline(x=x, c="red")

        plt.legend()
        plt.title(f"Training {title}")
        plt.savefig("imgs/train_ai.png")
        plt.close()


def rgb2grayscale(rgb: np.ndarray) -> np.ndarray:
    """Transform RGB image to grayscale

    Args:
        rgb (np.ndarray): RGB image to transform

    Returns:
        np.ndarray: Grayscale image
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

    def __init__(
        self,
        memory_capacity: int,
        train_buffer_capacity: int,
        test_buffer_capacity: int,
    ) -> None:
        """Initialization of the replay memory

        Args:
            memory_capacity (int): Maximum number of elements to fit in the memory
            train_buffer_capacity (int): Maximum number of elements to fit in the train buffer
            test_buffer_capacity (int): Maximum number of elements to fit in the test buffer
        """
        self.memory_capacity = memory_capacity
        self.train_buffer_capacity = train_buffer_capacity
        self.test_buffer_capacity = test_buffer_capacity
        self.memory = []
        self.train_buffer = []
        self.test_buffer = []
        self.memory_position = 0

    def push_to_memory(self, *args) -> None:
        """Save a transition to memory"""
        if len(self.memory) < self.memory_capacity:
            self.memory.append(None)
        self.memory[self.memory_position] = Transition(*args)
        self.memory_position = (self.memory_position + 1) % self.memory_capacity

    def push_to_train_buffer(self, *args) -> None:
        """Save a transition to train buffer"""
        self.train_buffer.append(Transition(*args))
        if len(self.train_buffer) > self.train_buffer_capacity:
            raise Exception("Error: capacity of the train_buffer exceded")

    def push_to_test_buffer(self, ob: np.ndarray) -> None:
        """Save an observation to test buffer

        Args:
            ob (np.ndarray): Observation/state to push into the buffer
        """
        self.test_buffer.append(ob)
        if len(self.test_buffer) > self.test_buffer_capacity:
            raise Exception("Error: capacity of the test_buffer exceded")

    def sample(self, batch_size: int) -> np.ndarray:
        """Sample batch_size random elements from memory

        Args:
            batch_size (int): Number of elements to sample

        Returns:
            np.ndarray: Sampled elements
        """
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        """Overwrite of the len function for the object

        Returns:
            int: Length of the memory
        """
        return len(self.memory)