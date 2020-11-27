from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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
        self.fc1 = nn.Linear(3872, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, action_space_dim)

        # initialization weights
        torch.nn.init.normal_(self.cnv1.weight)
        torch.nn.init.normal_(self.cnv2.weight)
        torch.nn.init.normal_(self.fc1.weight)
        torch.nn.init.normal_(self.fc2.weight)

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
        player_id: int = 1,
        name: str = "\(°_°')/",
        batch_size: int = 128,
        gamma: float = 0.98,
        memory_size: int = 40000,
    ) -> None:
        """
        Initialization of the Agent
        ------------
        # Parameters
        ------------

        env : environment in which the agent operates, must be Wimblepong \n
        player_id : id assigned to the player, id determines on which side the ai is going to play
        """
        # list of parameters of the agent
        self.player_id = player_id
        self.name = name
        self.batch_size = batch_size  # size of batch for update
        self.gamma = gamma  # discount factor
        self.memory_size = memory_size  # size of replay memory
        self.memory = ReplayMemory(
            self.memory_size, train_buffer_capacity=4, test_buffer_capacity=4
        )

        # networks
        self.policy_net = DQN(action_space_dim=3, hidden_dim=256).to(
            torch.device(device)
        )
        self.target_net = DQN(action_space_dim=3, hidden_dim=256).to(
            torch.device(device)
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)

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
        non_final_mask = 1 - torch.tensor(batch.done, dtype=torch.uint8).to(
            torch.device(device)
        )
        non_final_mask = non_final_mask.type(torch.bool)
        non_final_next_obs = torch.stack(
            [ob for nonfinal, ob in zip(non_final_mask, batch.next_ob) if nonfinal]
        ).to(torch.device(device))
        ob_batch = torch.stack(batch.ob).to(torch.device(device))
        rew_batch = torch.stack(batch.rew).to(torch.device(device))
        action_batch = torch.stack(batch.action).to(torch.device(device))

        # estimate Q(st, a) with the policy network
        state_action_values = (
            self.policy_net.forward(ob_batch).gather(1, action_batch).squeeze()
        )

        # estimate V(st+1) with target network
        next_state_values = torch.zeros(self.batch_size).to(torch.device(device))
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
            param.grad.data.clamp_(-0.1, 0.1)
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
            # get stack of obeservations
            if train:
                ob_stack = self.get_stack_from_train_buffer(ob)
            else:
                ob_stack = self.get_stack_from_test_buffer(ob)
            ob_stack = ob_stack.unsqueeze(0)

            # predict best action
            with torch.no_grad():
                action = self.policy_net.forward(ob_stack).argmax().item()

        if not train:
            self.push_to_test_buffer(ob)

        return action

    def get_name(self) -> str:
        """
        Interface function to retrieve the agents name
        """
        return self.name

    def reset(self) -> None:
        """
        clean the buffers of the memory
        """
        self.memory.test_buffer = []
        self.memory.train_buffer = []

    def load_model(self, path: str = "weights/DQN_baseline.ai") -> None:
        """
        Load model from file
        """
        # TODO: change path before sending
        self.policy_net.load_state_dict(
            torch.load(path, map_location=torch.device(device))
        )
        self.policy_net.eval()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def push_to_train_buffer(
        self, ob: np.ndarray, action: int, reward: int, next_ob: np.ndarray, done: bool
    ) -> Any:
        """
        Push a Transition to the memory train buffer
        """
        # preprocess observations
        ob = self.preprocess_ob(ob)
        next_ob = self.preprocess_ob(next_ob)

        # save to buffer
        action = torch.Tensor([action]).long().to(torch.device(device))
        reward = torch.tensor([reward], dtype=torch.float32).to(torch.device(device))
        self.memory.push_to_test_buffer(ob, action, next_ob, reward, done)

        # check if I need to push to memory
        if len(self.memory.train_buffer) == self.memory.buffer_capacity or done:

            # get the buffer and transition elements to push into memory
            buffer = self.memory.train_buffer
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
                self.memory.train_buffer = self.memory.train_buffer[1:]

            # if done reset everything
            if done:
                self.reset()

    def push_to_test_buffer(self, ob: np.ndarray) -> Any:
        """
        Push a Transition to the test buffer
        """
        # preprocess observation and push to test buffer
        ob = self.preprocess_ob(ob)
        self.memory.push_to_test_buffer(ob)

        # check if I have filled it
        if len(self.memory.test_buffer) == self.memory.test_buffer_capacity:
            self.memory.test_buffer = self.memory.test_buffer[1:]

    def get_stack_from_train_buffer(self, ob) -> Tensor:
        """
        Get tensor of stacked observations to predict an action from train buffer
        """
        ob = self.preprocess_ob(ob)

        # get observations from train buffer
        obs = (
            [x.ob for x in self.memory.train_buffer]
            if len(self.memory.train_buffer) != 0
            else [ob]
        )
        obs.append(ob)

        # complete the sequence
        while len(obs) != self.memory.train_buffer_capacity:
            obs.append(obs[-1])

        # stack observations and return them
        ob_stack = torch.stack(obs).to(torch.device(device))

        return ob_stack

    def get_stack_from_test_buffer(self, ob: np.ndarray) -> Tensor:
        """
        Get tensor of stacked observations to predict an action from train buffer
        """
        ob = self.preprocess_ob(ob)

        # get observations from test buffer
        obs = (
            [x for x in self.memory.test_buffer]
            if len(self.memory.test_buffer) != 0
            else [ob]
        )
        obs.append(ob)

        # complete the sequence
        while len(obs) != self.memory.test_buffer_capacity:
            obs.append(obs[-1])

        # stack observations and return them
        ob_stack = torch.stack(obs).to(torch.device(device))

        return ob_stack

    def preprocess_ob(self, ob: np.ndarray) -> Tensor:
        """
        Preprocess image:
        - shrink the image to 100x100
        - transform it to black and white
        - transform it into a Tensor
        """
        # shrink image
        ob = Image.fromarray(ob)
        ob = ob.resize((100, 100))
        ob = np.asarray(ob)

        # grayscale image
        ob = rgb2grayscale(ob)
        ob[ob != ob[0][0]] = 1
        ob[ob == ob[0][0]] = 0

        # Tensor definition
        ob = torch.from_numpy(ob).float().to(torch.device(device))

        return ob