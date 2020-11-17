import argparse
import sys
import os
from pong_testbench import PongTestbench
from multiprocessing import Process, Queue
from matplotlib import font_manager
from time import sleep
import importlib
import traceback
import numpy as np
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("dir", type=str, help="Directory with agents.")
parser.add_argument("--render", "-r", action="store_true", help="Render the competition.")
parser.add_argument("--games", "-g", type=int, default=100, help="Number of games.")
parser.add_argument("--max_proc", "-p", type=int, default=4, help="Max number of processes.")
parser.add_argument("--start-file", "-f", type=str, default=None, help="Start file")

args = parser.parse_args()

save_file = "ebr_save.p"

def run_test(id1, agent1_dir, id2,  agent2_dir, queue, games, render):
    # Add the first agent to Python import path
    sys.path.insert(0, agent1_dir)
    orig_wd = os.getcwd()

    # Import the first agent
    try:
        import agent
    except Exception as e:
        print(f"!!! Something went wrong in {id1}:{id2} while importing 1st agent")
        print(f"!!! agent1_dir={agent1_dir}, agent2_dir={agent2_dir}")
        print(f"!!! Error")
        print("!!!", e)
        print("!!! Traceback")
        traceback.print_exc()
        return

    # chdir to the directory (needed for loading the model)
    # and instantiate the agent
    os.chdir(agent1_dir)
    try:
        agent1 = agent.Agent()
        agent1.load_model()
    except Exception as e:
        print(f"!!! Something went wrong in {id1}:{id2} while loading 1st agent")
        print(f"!!! agent1_dir={agent1_dir}, agent2_dir={agent2_dir}")
        print(f"!!! Error")
        print("!!!", e)
        print("!!! Traceback")
        traceback.print_exc()
        return

    # Go back to the original directory
    os.chdir(orig_wd)

    # Remove agent1 from path
    del sys.path[0]

    # Add the 2nd agent to path
    sys.path.insert(0, agent2_dir)

    # reload the agent module using agent.py from the new dir
    try:
        importlib.reload(agent)
    except Exception as e:
        print(f"!!! Something went wrong in {id1}:{id2} while importing 2nd agent")
        print(f"!!! agent1_dir={agent1_dir}, agent2_dir={agent2_dir}")
        print(f"!!! Error")
        print("!!!", e)
        print("!!! Traceback")
        traceback.print_exc()
        return

    # chdir, instantiate, cleanup (same as before)
    os.chdir(agent2_dir)
    try:
        agent2 = agent.Agent()
        agent2.load_model()
    except Exception as e:
        print(f"!!! Something went wrong in {id1}:{id2} while loading 2nd agent")
        print(f"!!! agent1_dir={agent1_dir}, agent2_dir={agent2_dir}")
        print(f"!!! Error")
        print("!!!", e)
        print("!!! Traceback")
        traceback.print_exc()
        return

    os.chdir(orig_wd)
    del sys.path[0]

    # Get names
    name1 = agent1.get_name()
    name2 = agent2.get_name()

    # Create and init the testbench for the agents
    testbench = PongTestbench(render)
    testbench.init_players(agent1, agent2)

    # Run the match
    try:
        testbench.run_test(games)
    except Exception as e:
        print(f"!!! Something went wrong in {name1} ({id1}) vs {name2} ({id2})")
        print(f"!!! Error")
        print("!!!", e)
        print("!!! Traceback")
        traceback.print_exc()
        return

    # Get scores and pass them to the parent process
    wins1, games = testbench.get_agent_score(agent1)
    wins2, games = testbench.get_agent_score(agent2)

    print(f"{name1} vs {name2} finished, wins1={wins1}, wins2={wins2}")

    queue.put((id1, id2, wins1, wins2, name1, name2, games))


def get_directories(top_dir):
    subdir_list = []
    subdir_gen = os.walk(top_dir)
    # Recursively scout the directory for agents
    for dir, subdirs, files in subdir_gen:
        if "__pycache__" in dir:
            continue
        if "agent.py" not in files:
            print("Warn: No agent.py found in %s. Skipping." % dir)
            continue
        subdir_list.append(dir)
        print("%s added to directory list." % dir)
    subdir_list.sort()

    # Return a list of folders with agent.py
    return subdir_list


def epic_battle_royale(top_dir, max_proc=4):
    directories = get_directories(top_dir)
    names = ["__unknown__"] * len(directories)
    procs = []
    result_queue = Queue()
    all_results = []
    skipdict = []
    print("Finished scanning for agents; found:", len(directories))

    if args.start_file is not None:
        with open(args.start_file, "rb") as f:
            all_results = pickle.load(f)
        for id1, id2, wins1, wins2, name1, name2, games in all_results:
            print(f"Skipping {name1}:{name2} cause already played")
            skipdict.append((id1, id2))
        print(f"Total skipped: {len(skipdict)}")

    for i1, d1 in enumerate(directories):
        for i2, d2 in enumerate(directories):
            if i1 == i2:
                continue
            if (i1, i2) in skipdict:
                continue
            pargs = (i1, d1, i2, d2, result_queue, args.games, args.render)
            proc = Process(target=run_test, args=pargs)
            procs.append(proc)
            print("Living procs:", sum(p.is_alive() for p in procs))
            while sum(p.is_alive() for p in procs) >= max_proc:
                sleep(0.3)
            print("Starting process (%d / %d)" % (i1*len(directories) + i2, len(directories)**2))
            proc.start()
            sleep(1)

            # Join dead ones
            new_p = []
            for p in procs:
                if not p.is_alive():
                    p.join(1)
                else:
                    new_p.append(p)
            procs = new_p

            while result_queue.qsize() > 0:
                all_results.append(result_queue.get())
            with open(save_file, "wb") as f:
                pickle.dump(all_results, f)

    for p in procs:
        try:
            # Give it some final timeout. 20 sec/game is a very safe choice.
            # It shouldn't happen anyway; it's there just to prevent us from
            # losing all results in case of some pipes issues or a deadlock
            timeout = args.games * 20
            p.join(timeout)
            p.terminate()

            # Prevent errors in old Python versions
            if hasattr(p, "kill"):
                p.kill()
        except Exception as e:
            print("Join/Terminate/Kill error")
            traceback.print_exc()
    while result_queue.qsize() > 0:
        all_results.append(result_queue.get())

    # Fetch all results from the queue
    no_agents = len(directories)
    games_won = np.zeros((no_agents, no_agents), dtype=np.int32)
    total_games = np.zeros((no_agents, ), dtype=np.int32)

    for id1, id2, wins1, wins2, name1, name2, games in all_results:
        # Sanity check...
        if wins1 + wins2 != games:
            print(f"Wins dont sum up! {name1} vs {name2}: {wins1}+{wins2} != {games}")
        games_won[id1, id2] += wins1
        games_won[id2, id1] += wins2
        names[id1] = name1
        names[id2] = name2
        total_games[id1] += games
        total_games[id2] += games

    # Save raw results as numpy
    np.save("brres", games_won)

    # Format: Wins of ROW versus COLUMN
    np.savetxt("battle_royale_results.txt", games_won, fmt="%d")
    np.savetxt("battle_royale_players.txt", directories, fmt="%s")

    # Sum across columns to get total wins of each agent
    total_wins = games_won.sum(axis=1)

    # And across rows to get total losses.
    total_losses = games_won.sum(axis=0)
    agent_wins = list(zip(total_wins, total_losses, names, directories, total_games))
    agent_wins.sort(key=lambda x: -x[0])

    # Save the leaderboard
    resfile = open("leaderboard.txt", "w")
    print("")
    print("-"*80)
    print("--- LEADERBOARD ---")
    for i, (wins, losses, name, dir, games) in enumerate(agent_wins):
        winrate = wins/(wins+losses)
        line = f"{i+1}. {name} with {wins} wins in {games} games (winrate {winrate*100:.2f}%) (from {dir})"
        resfile.write(line+"\n")
        print(line)
    resfile.close()
    print("-"*80)
    print("")

    print("Finished!")


if __name__ == "__main__":
    epic_battle_royale(args.dir, args.max_proc)
