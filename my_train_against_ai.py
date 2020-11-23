import re
import matplotlib.pyplot as plt
from random import randint
import pickle
import gym
import numpy as np
import argparse
import wimblepong
import my_agent
import my_utils
import torch
from PIL import Image

# CONFIGURATION VARIABLE
TARGET_UPDATE = 5  # update target_net every TARGET_UPDATE frames
FRAME_STALL = 4  # update policy_net every FRAME_STALL frames

# args parser
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--fps", type=int, help="FPS for rendering", default=30)
parser.add_argument("--scale", type=int, help="Scale of the rendered game", default=1)
args = parser.parse_args()

# Make the environment
env = gym.make("WimblepongVisualMultiplayer-v0")
env.unwrapped.scale = args.scale
env.unwrapped.fps = args.fps
episodes = 100000  # Number of episodes/games to play

# Define the player IDs for both agents
player = my_agent.Agent(env, 1)
opponent = wimblepong.SimpleAi(env, 2)
# player = wimblepong.SimpleAi(env, 1)

# Set the names for both SimpleAIs
env.set_names(player.get_name(), opponent.get_name())

# start training
wins = [0]
for ep in range(0, episodes):

    done = False
    (ob, _) = env.reset()
    frame = 0
    while not done:

        # Get the actions from both SimpleAIs
        action1 = player.get_action(ob)
        action2 = opponent.get_action()

        # Step the environment and get the rewards and new observations
        (next_ob, _), (rew, _), done, info = env.step((action1, action2))

        # update agent policy
        player.push_to_memory(ob, action1, rew, next_ob, done)
        frame += 1
        if frame % FRAME_STALL == 0:
            player.update_policy_net()

        # move to next observation
        ob = next_ob

        # Count the wins
        if rew != 0:
            wins.append(1) if rew == 10 else wins.append(0)

        # render the frames
        if not args.headless:
            env.render()

    # Update training image
    print(f"Episode {ep+1} finised")
    if (ep + 1) % 10 == 0:
        my_utils.plot_winsratio(wins, "Training process")

    # update target_net
    if (ep + 1) % TARGET_UPDATE == 0:
        player.update_target_net()

    # Save the policy
    if (ep + 1) % 1000 == 0:
        torch.save(player.policy_net.state_dict(), "imgs/DQN_weights.ai")
