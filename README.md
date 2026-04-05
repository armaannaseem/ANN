# DQN GridWorld

A Deep Q-Network (DQN) implemented in pure Python and NumPy, with no ML 
libraries. Trained on a custom GridWorld environment to demonstrate 
reinforcement learning fundamentals from first principles.

## What this is

A reinforcement learning agent that learns to navigate a 4x4 grid from a 
start position to a goal while avoiding walls. The agent has no prior 
knowledge of the environment; it learns purely through trial and error, 
guided by reward signals and the Bellman optimality equation.

The neural network is implemented entirely from scratch: forward pass, 
backpropagation, and gradient descent are all written as explicit NumPy 
matrix operations with no ML framework involvement.

## Architecture

**Neural network** (`neural_network.py`)
- 2 inputs (agent x, y position)
- 64 hidden neurons with ReLU activation
- 4 outputs (Q-values for up, down, left, right)
- Trained via batched backpropagation and gradient descent

**Experience Replay** (`replay_buffer.py`)
- Stores up to 10,000 recent agent experiences `(state, action, reward, next_state, done)`
- Random sub-samples of 32 experiences are drawn for training to break temporal correlations and stabilize learning.

**Environment** (`gridworld.py`)
- 4x4 grid with a fixed goal at (3,3) and three walls
- Rewards: +1 for reaching goal, -1 for hitting a wall, 0 otherwise
- Episode terminates on goal reached or 200 steps

**Training loop** (`train.py`)
- Epsilon-greedy exploration decaying from 1.0 to 0.1 over 1000 episodes
- **Target Network:** Employs a secondary "frozen" neural network to calculate stable Bellman Q-value targets. This network synchronizes with the main network every 10 episodes.
- Bellman equation generates Q-value targets for each batch:
  `Q(s,a) = reward + γ × max Q_target(s', a')`
- Discount factor γ = 0.99, learning rate = 0.01

## Results

The agent begins with fully random behaviour (episode 0: ~-15 total reward) 
and converges to a near-optimal policy around episode 300-400 (total reward: 1), 
demonstrating successful learning from scratch using batched matrix multiplications.

## How to run

Requires Python 3 and NumPy only.
```bash
pip install numpy
python3 train.py
```

## Why no ML libraries

The goal was to demonstrate that every component (weight initialisation, 
the forward pass, batched backpropagation via the chain rule, and the Bellman update) can be expressed as explicit matrix operations.
Using PyTorch or TensorFlow would abstract away precisely the parts this project is meant to show.