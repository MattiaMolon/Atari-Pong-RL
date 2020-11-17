import argparse
import sys
import os
from pong_testbench import PongTestbench
from multiprocessing import Process
from matplotlib import font_manager
from time import sleep

parser = argparse.ArgumentParser()
parser.add_argument("dir", type=str, help="Directory with agents.")
parser.add_argument("--render", "-r", action="store_true", help="Render the competition.")
parser.add_argument("--games", "-g", type=int, default=100, help="Number of games.")
parser.add_argument("--max_proc", "-p", type=int, default=4, help="Max number of processes.")

args = parser.parse_args()


def save_winrate(dir, wins, games):
    resfile = open(os.path.join(dir, "simpleai_winrate.txt"), "w")
    resfile.write("Winrate: %f (%d / %d" % (wins/games, wins, games))
    resfile.close()


def run_test(agent_dir, games, render):
    sys.path.insert(0, agent_dir)
    orig_wd = os.getcwd()
    from agent import Agent as Agent1
    os.chdir(agent_dir)
    agent1 = Agent1()
    agent1.load_model()
    os.chdir(orig_wd)
    del sys.path[0]

    testbench = PongTestbench(render, silent=False)
    testbench.init_players(agent1, None)
    testbench.run_test(games)

    wins, games = testbench.get_agent_score(agent1)
    save_winrate(agent_dir, wins, games)


def get_directories(top_dir):
    subdir_list = []
    subdir_gen = os.walk(top_dir)
    for dir, subdirs, files in subdir_gen:
        if "__pycache__" in dir:
            continue
        if "agent.py" not in files:
            print("Warn: No agent.py found in %s. Skipping." % dir)
            continue
        subdir_list.append(dir)
        print("%s added to directory list." % dir)
    return subdir_list


def mass_test(top_dir, max_proc=4):
    directories = get_directories(top_dir)
    procs = []
    print("Finished scanning for agents; found:", len(directories))

    for d in directories:
        proc = Process(target=run_test, args=(d, args.games, args.render))
        procs.append(proc)
        print("Living procs:", sum(p.is_alive() for p in procs))
        while sum(p.is_alive() for p in procs) >= max_proc:
            sleep(0.3)
            print(".", end="", flush=True)
        print("Starting process")
        proc.start()
        sleep(1)

    for p in procs:
        p.join()

    print("Finished!")


if __name__ == "__main__":
    mass_test(args.dir, args.max_proc)
