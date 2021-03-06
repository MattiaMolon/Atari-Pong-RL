# DQN agent for Wimblepong

This project is a DQN agent able to play in the pong-v0 OpenAI Gym environment.
The environment is used in the Reinforcement Learning course at Aalto University,
Finland.

<p align="center">
  <img width="300" height="350" src="./submission/game.gif">
</p>

## :wrench: How to use
- Clone the repository: `https://github.com/MattiaMolon/Atari-Pong-RL`.
- Install the dependencies listed in requirements.txt.
- **to train:** check the `train_basic.py` to train the agent against simpleAI or `train_hibrid.py` to train the agent one game against simpleAI and one game against himself (This helps the agent to not overfit against simpleAI alone). Run both files with the flag `--train True`.
- **to test:** check the `train_basic.py` to test the agent against simoleAI or `train_hibrid.py` to test the agent against another agent. Run both the files **without** the flag `--train True`. 

## :robot: Agents
 - The **SimpleAI** agent is an agent that uses the absolute ball and player positions to follow the ball and reflect it in random directions.
 - The agent implemented in this repository consists in a **DQN agent** with experience replay and can be found in `agent.py`. The weights in `./weights/hibrid_tuned_best.ai` can reach a winrate of 80% against simpleAI.

## :scroll: Report
The report `pdf_files/RL_final_project.pdf` includes all the details about the implementation of the agent and the training procedures followed to reach the aformentioned scores. 

## :earth_africa: Environment
Additional details on the environment used and how to interact with it are available in the [official repository](https://github.com/aalto-intelligent-robotics/wimblepong) of the course.

## :stars: Future works:
- [x] implementation DQN
- [x] implementation experience replay, buffers, and preprocessing
- [ ] implementation of PPO
- [ ] implementation of actor-critic
