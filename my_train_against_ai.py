import matplotlib.pyplot as plt
from random import randint
import pickle
import gym
import numpy as np
import argparse
import wimblepong
import my_agent
import my_utils
from PIL import Image

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
# player = my_agent.Agent(env, 1)
opponent = wimblepong.SimpleAi(env, 2)
player = wimblepong.SimpleAi(env, 1)

# Set the names for both SimpleAIs
env.set_names(player.get_name(), opponent.get_name())

# start training
wins = [0]
for i in range(0, episodes):

    done = False
    while not done:

        # Get the actions from both SimpleAIs
        action1 = player.get_action()
        action2 = opponent.get_action()

        # Step the environment and get the rewards and new observations
        (ob1, ob2), (rew1, rew2), done, info = env.step((action1, action2))

        # Count the wins
        if rew1 != 0:
            wins.append(1) if rew1 == 10 else wins.append(0)

        # render the frames
        if not args.headless:
            env.render()

        # reset env and
        if done:
            observation = env.reset()
            print(f"Episode {i} finised")
            if i % 5 == 0:
                my_utils.plot_winsratio(wins, "Training process")