# DQN agent for Wimblepong

This project is a DQN agent able to play in the pong-v0 OpenAI Gym environment.
The environment is used in the Reinforcement Learning course at Aalto University,
Finland.

## How to use
- Clone the repository: `https://github.com/MattiaMolon/Atari-Pong-RL`.
- Install the dependencies listed in requirements.txt.
- **to train:** check the `train_basic.py` to train the agent against simpleAI or `train_hibrid.py` to train the agent one game against simpleAI and one game against himself (This helps the agent to not overfit against simpleAI alone). Run both files with the flag `--train True`.
- **to test:** check the `train_basic.py` to test the agent against simoleAI or `train_hibrid.py` to test the agent against another agent. Run both the files **without** the flag `--train True`. 

## DQN agent
The agent implemented in this repository consists in a DQN agent with experience replay and can be found in `agent.py`. The weights in the `./weights` folder can reach a winrate of 80% against simpleAI.

## SimpleAI
The SimpleAI agent is an agent that uses the absolute ball and player positions to follow the ball and reflect it in random directions.

## Report
The report including all the details about the implementation of the agent and the training procedured followed to reach the aformentioned scores will be soon uploaded.

## Environment
Additional details on the environment used and how to interact with it are available in the [official repository](https://github.com/aalto-intelligent-robotics/wimblepong) of the course.

## Future works:
- implementation of PPO
- implementation of actor-critic
