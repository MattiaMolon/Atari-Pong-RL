import gym
import argparse
import wimblepong
import agent
import Ugo_utils


# CONFIGURATION VARIABLE
TARGET_UPDATE = 50  # update target_net every TARGET_UPDATE frames
GLIE_A = 10000  # a paramenter in glie -> a = 10000 means eps = 0.5 when episode = 10000
SAVE_POLICY_TIME = 500  # epochs between saves of the policy
SAVE_PLOT_TIME = 20  # epochs between saves of the plot
START_EPISODE = 31000  # episode from which to start training again. Should be 0 if starting training from scratch


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
player = agent.Agent(1)
player.load_model()
opponent = wimblepong.SimpleAi(env, 2)


# Set the names for both SimpleAIs
env.set_names(player.get_name(), opponent.get_name())

# start training
wins = []
for ep in range(START_EPISODE, episodes):

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
        if args.train:
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
        print(f"Episode {ep+1} finised, \t epsilon used = {round(epsilon, 6)}")

        # Update training image
        if (ep + 1) % SAVE_PLOT_TIME == 0:
            Ugo_utils.plot_winsratio(wins, "DQN with experience replay", START_EPISODE)

        # update target_net
        if (ep + 1) % TARGET_UPDATE == 0:
            player.update_target_net()

        # Save the policy
        if (ep + 1) % SAVE_POLICY_TIME == 0:
            player.save_model(dir="weights", ep=ep)