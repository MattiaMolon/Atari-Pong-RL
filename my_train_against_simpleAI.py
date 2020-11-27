import gym
import argparse
import wimblepong
import my_agent
import my_utils
import torch


# CONFIGURATION VARIABLE
TARGET_UPDATE = 50  # update target_net every TARGET_UPDATE frames
GLIE_A = 10000  # a paramenter in glie -> a = 10000 means eps = 0.5 when episode = 20000
SAVE_POLICY_TIME = 500  # epochs between saves of the policy
SAVE_PLOT_TIME = 20  # epochs between saves of the plot
START_UPDATE = 2000  # minimum number of elements in memory before starting updating. Should be 0 if not loading a previous model


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
# TODO: fix train boolean
parser.add_argument(
    "--train", "--t", type=bool, help="decide if train the model or not", default=False
)
args = parser.parse_args()


# Make the environment
env = gym.make("WimblepongVisualMultiplayer-v0")
env.unwrapped.scale = args.scale
env.unwrapped.fps = args.fps
episodes = 100000  # Number of episodes/games to play


# Define the player IDs for both agents
player = my_agent.Agent(1)
opponent = wimblepong.SimpleAi(env, 2)


# Set the names for both SimpleAIs
env.set_names(player.get_name(), opponent.get_name())

# start training
# load weights if requested
start_ep_train = 0
if args.load is not None:
    player.load_model(args.load)

    if args.train:
        # restart from baseline episode
        start_ep_train = int(args.load.split("_")[-1].split(".")[0])

wins = [0]
for ep in range(start_ep_train, episodes):

    done = False
    (ob, _) = env.reset()
    epsilon = GLIE_A / (GLIE_A + ep)
    epsilon = 0.05 if epsilon < 0.05 else epsilon
    while not done:

        # Get the actions from both SimpleAIs
        action1 = player.get_action(ob, epsilon, args.train)
        action2 = opponent.get_action()

        # Step the environment and get the rewards and new observations
        (next_ob, _), (rew, _), done, info = env.step((action1, action2))

        # update agent policy
        if args.train and len(player.memory.memory) >= START_UPDATE:
            player.push_to_train_buffer(ob, action1, rew, next_ob, done)
            player.update_policy_net()

        # move to next observation
        ob = next_ob

        # Count the wins
        if rew != 0:
            wins.append(1) if rew == 10 else wins.append(0)

        # render the frames
        if not args.headless:
            env.render()

    if args.train:
        print(f"Episode {ep+1} finised")

        # Update training image
        if (ep + 1) % SAVE_PLOT_TIME == 0:
            my_utils.plot_winsratio(wins, "DQN with experience replay", start_ep_train)

        # update target_net
        if (ep + 1) % TARGET_UPDATE == 0 and len(player.memory.memory) >= START_UPDATE:
            player.update_target_net()

        # Save the policy
        if (ep + 1) % SAVE_POLICY_TIME == 0:
            torch.save(player.policy_net.state_dict(), f"weights/DQN_{ep+1}.ai")
