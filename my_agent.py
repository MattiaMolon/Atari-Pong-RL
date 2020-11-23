import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import my_utils
from wimblepong import Wimblepong


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

    def __init__(self, action_space_dim=3, hidden_dim=256):
        """
        Initialization of the DQN
        ------------
        # Parameters
        ------------

        action_space_dim : dimension of the action space (0: stay still, 1: go up, 2: do down)
        hidden_dim : dimension of the embedding space of the input image
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
        Forward the image in the network
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


class Agent(object):
    """
    RL agent for the Atari game
    ---------------
    # Functions
    ---------------

    __init__() : initialize the agent
    get_name() : get name for the agent
    get_action() : get action to take given a frame as input
    reset() : reset state for the agent
    """

    def __init__(self, env, player_id=1):
        """
        Initialization of the Agent
        ------------
        # Parameters
        ------------

        env : environment in which the agent operates, must be Wimblepong
        player_id : id assigned to the player, id determines on which side the ai is going to play
        """
        # check if environment is correct
        if type(env) is not Wimblepong:
            raise TypeError("I'm not a very smart AI. All I can play is Wimblepong.")

        # list of parameters of the agent
        self.env = env
        self.player_id = id
        self.name = "\(°_°')/"
        self.policy_net = DQN(action_space_dim=3, hidden_dim=256)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.batch_size = 32  # size of batch for update
        self.gamma = 0.98  # discount factor

    def get_action(self, ob=None):
        """
        Interface function that returns the action that the agent took based
        on the observation ob

        ------------
        # Parameters
        ------------

        ob : frame from the game
        """
        # grayscale image
        ob_gray = my_utils.rgb2grayscale(ob)
        ob_gray = np.expand_dims(ob_gray, 0)
        ob_gray = torch.Tensor(ob_gray)

        # forward
        # TODO: epsilon greedy
        action = self.policy_net.forward(ob_gray).argmax().item()

        return action

    def get_name(self):
        """
        Interface function to retrieve the agents name
        """
        return self.name

    def reset(self):
        """
        ???
        """
        # Nothing to done for now...
        return