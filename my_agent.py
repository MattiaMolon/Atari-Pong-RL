import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# DQN architecture
class DQN(nn.Module):
    """
    Defines the structure of the DQN architecture used to train the agent
    ---------------
    # Functions
    ---------------

    __init__() : initialize the class
    forward() : compute forward pass in the network
    """

    def __init__(self, action_space_dim, hidden_dim=256):
        """
        initialization of the DQN
        ------------
        # Parameters
        ------------

        action_space_dim : dimension of the action space
        emd_dim : dimension of the embedding space of the input image
        """

        # call to super
        super(DQN, self).__init__()

        # parameters of the network
        self.hidden_dim = hidden_dim  # embedding space

        # DNN architecture
        # cnv -> convolutional
        # fc -> fully connected
        self.cnv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=8)
        self.cnv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4)
        self.cnv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4)
        self.flat1 = nn.Flatten()
        self.fc1 = nn.Linear(64 * 21 * 21, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, action_space_dim)

    def forward(self, x):
        """
        forward the image in the network
        ------------
        # Parameters
        ------------

        x : input image to feed forward into the network
        """
        x = F.max_pool2d(F.relu(self.cnv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.cnv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.cnv3(x)), (2, 2))
        x = self.flat1(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
