import argparse
import sys
import os
from pong_testbench import PongTestbench
from matplotlib import font_manager
import importlib

parser = argparse.ArgumentParser()
parser.add_argument("dir1", type=str, help="Directory to agent 1 to be tested.")
parser.add_argument("dir2", type=str, default=None, nargs="?",
                    help="Directory to agent 2 to be tested. If empty, SimpleAI is used instead.")
parser.add_argument("--render", "-r", action="store_true", help="Render the competition.")
parser.add_argument("--games", "-g", type=int, default=100, help="number of games.")

args = parser.parse_args()

sys.path.insert(0, args.dir1)
import agent
orig_wd = os.getcwd()
os.chdir(args.dir1)
agent1 = agent.Agent()
agent1.load_model()
os.chdir(orig_wd)
del sys.path[0]

if args.dir2:
    sys.path.insert(0, args.dir2)
    importlib.reload(agent)
    os.chdir(args.dir2)
    agent2 = agent.Agent()
    agent2.load_model()
    os.chdir(orig_wd)
    del sys.path[0]
else:
    agent2 = None

testbench = PongTestbench(args.render, silent=False)
testbench.init_players(agent1, agent2)
testbench.run_test(args.games)
