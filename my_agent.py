import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import my_utils

from torch.tensor import Tensor
from my_utils import Transition, ReplayMemory
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
        self.cnv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=8)
        self.cnv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4)
        self.cnv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4)
        self.flat1 = nn.Flatten()
        self.fc1 = nn.Linear(32 * 21 * 21, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, action_space_dim)

    def forward(self, x: Tensor) -> Tensor:
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
        batch_size: int = 32,
        gamma: float = 0.98,
        memory_size: int = 10000,
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
        self.memory = ReplayMemory(self.memory_size)

        # networks
        self.policy_net = DQN(action_space_dim=3, hidden_dim=256)
        self.target_net = DQN(action_space_dim=3, hidden_dim=256)
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
        state_action_values = self.policy_net.forward(ob_batch).gather(1, action_batch)

        # estimate V(st+1) with target network
        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask] = (
            self.target_net.forward(non_final_next_obs).max(1)[0].detach()
        )

        if next_state_values.size != state_action_values.size:
            # fmt: off
            import IPython ; IPython.embed()
            # fmt: on

        # expected Q value
        expected_state_action_values = rew_batch + self.gamma * next_state_values

        # loss
        loss = F.smooth_l1_loss(
            state_action_values.squeeze(), expected_state_action_values
        )

        # optimize the network
        self.optimizer.zero_grad()
        loss.backward()
        # TODO: check if these values are limiting too much
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1e-1, 1e-1)
        self.optimizer.step()

    def update_target_net(self) -> None:
        """
        Update target net
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_action(self, ob: np.ndarray = None, epsilon: float = 0.1) -> int:
        """
        Interface function that returns the action that the agent took based
        on the observation ob
        ------------
        # Parameters
        ------------

        ob : frame from the game
        """
        # preprocess ob
        ob = self.preprocess_ob(ob, batch_form=True)

        # epsilon greedy action selection
        # TODO: use glie
        if np.random.rand() < epsilon:
            action = np.random.randint(0, 4)
        else:
            action = self.policy_net.forward(ob).argmax().item()

        return action

    def get_name(self) -> str:
        """
        Interface function to retrieve the agents name
        """
        return self.name

    def reset(self) -> None:
        """
        ???
        """
        # Nothing to do for now...
        return

    def load_model(self, path: str = "weights/DQN_weights.ai") -> None:
        """
        Load model from file
        """
        # TODO: change path before sending
        self.policy_net.load_state_dict(torch.load(path))
        self.policy_net.eval()

    def push_to_memory(self, ob, action, reward, next_ob, done):
        """
        Push a Transition to memory
        """
        # preprocess observations
        ob = self.preprocess_ob(ob, batch_form=False)
        next_ob = self.preprocess_ob(next_ob, batch_form=False)

        # save to memory
        action = torch.Tensor([action]).long()
        reward = torch.tensor([reward], dtype=torch.float32)
        next_ob = torch.from_numpy(next_ob).float()
        ob = torch.from_numpy(ob).float()
        self.memory.push(ob, action, next_ob, reward, done)

    def preprocess_ob(self, ob: np.ndarray, batch_form=True) -> Tensor:
        """
        Preprocess image(s)
        """
        # grayscale image
        ob_gray = my_utils.rgb2grayscale(ob, self.player_id)

        # transform into batch if requested
        if batch_form:
            ob_gray = np.expand_dims(ob_gray, 0)
            ob_gray = torch.Tensor(ob_gray)

        return ob_gray