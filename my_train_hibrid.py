import gym
import argparse
import wimblepong
import numpy as np
import my_agent
import my_utils
import torch.optim as optim


# CONFIGURATION VARIABLE
UPDATE_OPPONENT_TRSH = 0.6  # if the agent wins at least 60% of the games, the opponent is updated with the weights of the agent
UPDATE_OPPONENT_LAG = (
    300  # wait to update opponent policy since the last time it has been done
)
UPDATE_OPPONENT_WSIZE = (
    200  # window size to consider in order to conpute WR in selfplay
)
TARGET_UPDATE = 50  # update target_net every TARGET_UPDATE games
SAVE_POLICY_TIME = 500  # epochs between saves of the policy
SAVE_PLOT_TIME = 20  # epochs between saves of the plot
START_EPISODE = 0  # episode from which to start training again. Should be 0 if starting training from scratch
SWITCH_OPPONENT = 1  # number of games between swtich of opponents


# args parser
parser = argparse.ArgumentParser()
parser.add_argument("--fps", type=int, help="FPS for rendering", default=30)
parser.add_argument(
    "--headless", "--h", action="store_true", help="Run in headless mode"
)
parser.add_argument(
    "--scale", "--s", type=int, help="Scale of the rendered game", default=1
)
parser.add_argument(
    "--load", "--l", type=str, help="load model from file", default=None
)
parser.add_argument(
    "--train", "--t", type=bool, help="decide if train the model or not", default=False
)
args = parser.parse_args()
print("Train: ", args.train)


# Make the environment
env = gym.make("WimblepongVisualMultiplayer-v0")
env.unwrapped.scale = args.scale
env.unwrapped.fps = args.fps
episodes = 100000  # Number of episodes/games to play


# Define the player IDs for both agents
player = my_agent.Agent(1)
player.load_model(path_ai="weights/simpleAI_best.ai")
opponent = my_agent.Agent(2)
opponent.load_model(path_ai="weights/simpleAI_best.ai")


# Set the names for both SimpleAIs
env.set_names(player.get_name(), opponent.get_name())


# start training
wins = []
lag_opponent_update = 0
opponent_update_idxs = []
for ep in range(START_EPISODE, episodes):

    done = False
    (ob, _) = env.reset()
    epsilon = 0.1
    while not done:

        # Get the actions from both
        action1 = player.get_action(ob, epsilon, args.train)

        if opponent.get_name() == "\(°_°')/":
            action2 = opponent.get_action(np.fliplr(ob))
        else:
            action2 = opponent.get_action(ob)

        # Step the environment and get the rewards and new observations
        (next_ob, _), (rew, _), done, info = env.step((action1, action2))

        # update agent policy
        if args.train:
            player.push_to_train_buffer(ob, action1, rew, next_ob, done)
            player.update_policy_net()

        # move to next observation
        ob = next_ob

        # Count the wins
        if rew != 0:
            wins.append(1) if rew == 10 else wins.append(0)
            lag_opponent_update += 1

        # render the frames
        if not args.headless:
            env.render()

    if args.train:
        print(f"Episode {ep+1} finised")

        if (ep + 1) % SWITCH_OPPONENT == 0:
            if opponent.get_name() == "\(°_°')/":
                opponent = wimblepong.SimpleAi(env, 2)
            else:
                opponent = my_agent.Agent(1)
                opponent.load_model("weights/simpleAI_best.ai")

        # Update training image
        if (ep + 1) % SAVE_PLOT_TIME == 0:
            my_utils.plot_winsratio(
                wins,
                "DQN with experience replay",
                START_EPISODE,
                opponent_update_idxs=opponent_update_idxs,
            )

        # update target_net
        if (ep + 1) % TARGET_UPDATE == 0:
            player.update_target_net()

        # Save the policy
        if (ep + 1) % SAVE_POLICY_TIME == 0:
            player.save_model(dir="weights", ep=ep)