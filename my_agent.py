from math import exp
import random
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

from PIL import Image
from torch.tensor import Tensor
from my_utils import Transition, ReplayMemory, rgb2grayscale
from wimblepong import Wimblepong


# check for cuda
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


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

    def __init__(self, action_space_dim=3, hidden_dim=256) -> None:
        """
        Initialization of the DQN
        ------------
        # Parameters
        ------------

        action_space_dim : dimension of the action space (0: stay still, 1: go up, 2: do down)  \n
        hidden_dim : dimension of the embedding space of the input image
        """

        # call to super
        super(DQN, self).__init__()

        # parameters of the network
        self.hidden_dim = hidden_dim

        # DNN architecture
        # cnv -> convolutional
        # fc -> fully connected
        self.cnv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4)
        self.cnv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.flat1 = nn.Flatten()
        self.fc1 = nn.Linear(2592, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, action_space_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward the image in the network
        ------------
        # Parameters
        ------------

        x : input image to feed forward into the network
        """
        x = F.relu(self.cnv1(x))
        x = F.relu(self.cnv2(x))
        x = self.flat1(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# DQN Agent
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

    def __init__(
        self,
        env,
        player_id: int = 1,
        name: str = "\(°_°')/",
        batch_size: int = 64,
        gamma: float = 0.95,
        memory_size: int = 50000,
    ) -> None:
        """
        Initialization of the Agent
        ------------
        # Parameters
        ------------

        env : environment in which the agent operates, must be Wimblepong \n
        player_id : id assigned to the player, id determines on which side the ai is going to play
        """
        # check if environment is correct
        if type(env) is not Wimblepong:
            raise TypeError("I'm not a very smart AI. All I can play is Wimblepong.")

        # list of parameters of the agent
        self.env = env
        self.player_id = player_id
        self.name = name
        self.batch_size = batch_size  # size of batch for update
        self.gamma = gamma  # discount factor
        self.memory_size = memory_size  # size of replay memory
        self.memory = ReplayMemory(self.memory_size, 4)

        # networks
        self.policy_net = DQN(action_space_dim=3, hidden_dim=256).to(
            torch.device(device)
        )
        self.target_net = DQN(action_space_dim=3, hidden_dim=256).to(
            torch.device(device)
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)

    def update_policy_net(self) -> None:
        """
        Update policy network
        ------------
        # Parameters
        ------------

        TODO: params
        """
        # check if memory has enough elements to sample
        if len(self.memory) < self.batch_size:
            return

        # get transitions
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # get elements from batch
        non_final_mask = 1 - torch.tensor(batch.done, dtype=torch.uint8)
        non_final_mask = non_final_mask.type(torch.bool)
        non_final_next_obs = torch.stack(
            [ob for nonfinal, ob in zip(non_final_mask, batch.next_ob) if nonfinal]
        )
        ob_batch = torch.stack(batch.ob)
        rew_batch = torch.stack(batch.rew)
        action_batch = torch.stack(batch.action)

        # estimate Q(st, a) with the policy network
        state_action_values = (
            self.policy_net.forward(ob_batch).gather(1, action_batch).squeeze()
        )

        # estimate V(st+1) with target network
        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask] = (
            self.target_net.forward(non_final_next_obs).max(1)[0].detach()
        )

        # expected Q value
        expected_state_action_values = (
            rew_batch.squeeze() + self.gamma * next_state_values
        )

        # loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # optimize the network
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1e-1, 1e-1)
        self.optimizer.step()

    def update_target_net(self) -> None:
        """
        Update target net
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_action(
        self, ob: np.ndarray = None, epsilon: float = 0.1, train: bool = False
    ) -> int:
        """
        Interface function that returns the action that the agent took based
        on the observation ob
        ------------
        # Parameters
        ------------

        ob : frame from the game
        """

        # epsilon greedy action selection
        if train and np.random.rand() < epsilon:
            action = np.random.randint(0, 3)
        else:
            # preprocess ob
            ob = self.get_action_from_buffer(ob)
            ob = ob.unsqueeze(0)

            # predict best action
            action = self.policy_net.forward(ob).argmax().item()

        return action

    def get_name(self) -> str:
        """
        Interface function to retrieve the agents name
        """
        return self.name

    def reset(self) -> None:
        """
        clean the buffer of the memory
        """
        self.memory.buffer = []

    def load_model(self, path: str = "weights/DQN_weights.ai") -> None:
        """
        Load model from file
        """
        # TODO: change path before sending
        self.policy_net.load_state_dict(
            torch.load(path, map_location=torch.device(device))
        )
        self.policy_net.eval()

    def push_to_buffer(self, ob, action, reward, next_ob, done) -> Any:
        """
        Push a Transition to the memory buffer
        """
        # preprocess observations
        ob = self.preprocess_ob(ob)
        next_ob = self.preprocess_ob(next_ob)

        # save to buffer
        action = torch.Tensor([action]).long().to(torch.device(device))
        reward = torch.tensor([reward], dtype=torch.float32).to(torch.device(device))
        self.memory.push_to_buffer(ob, action, next_ob, reward, done)

        # check if I need to stack images
        if len(self.memory.buffer) == self.memory.buffer_capacity or done:

            # get the buffer and transition elements to push into memory
            buffer = self.memory.buffer
            ob_stack = torch.stack(
                (buffer[0].ob, buffer[1].ob, buffer[2].ob, buffer[3].ob)
            ).to(torch.device(device))
            next_ob_stack = torch.stack(
                (
                    buffer[0].next_ob,
                    buffer[1].next_ob,
                    buffer[2].next_ob,
                    buffer[3].next_ob,
                )
            ).to(torch.device(device))

            # push to memory
            self.memory.push_to_memory(
                ob_stack,
                buffer[3].action,
                next_ob_stack,
                buffer[3].rew,
                buffer[3].done,
            )

            # if not done delete the firt row in the buffer
            if not done:
                self.memory.buffer = self.memory.buffer[1:]

            # if done reset everything
            if done:
                self.reset()

    def get_action_from_buffer(self, ob) -> Tensor:
        """
        Get tensor of stacked observations to predict an action
        """
        # TODO: WE WILL NOT HAVE THE BUFFER DURING TESTING! FIX THIS

        # get observations from buffer
        obs = (
            [x.ob for x in self.memory.buffer] if len(self.memory.buffer) != 0 else [ob]
        )

        # I don't have filled the buffer yet
        if len(self.memory.buffer) < self.memory.buffer_capacity:
            # create new observations which do not exists
            while len(obs) != self.memory.buffer_capacity:
                obs.append(obs[-1])

        # stack observations and return them
        ob_stack = torch.stack(obs).to(torch.device(device))

        return ob_stack

    def preprocess_ob(self, ob: np.ndarray) -> Tensor:
        """
        Preprocess image:
        - shrink the image to 84x84
        - transform it to grayscale
        - transform it into a Tensor
        """
        # shrink image
        ob = Image.fromarray(ob)
        ob = ob.resize((84, 84))
        ob = np.asarray(ob)

        # grayscale image
        ob = rgb2grayscale(ob)

        # Tensor definition
        ob = torch.from_numpy(ob).float().to(torch.device(device))

        return ob