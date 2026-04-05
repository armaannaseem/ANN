import numpy as np
from neural_network import NeuralNetwork
from gridworld import GridWorld
from replay_buffer import ReplayBuffer

# initialising grid world and neural networks
env = GridWorld()
nn = NeuralNetwork(input_size=2, hidden_size=64, output_size=4)
target_nn = NeuralNetwork(input_size=2, hidden_size=64, output_size=4)
target_nn.copy_weights(nn) # Initial sync of target network

buffer = ReplayBuffer(10000)
batch_size = 32
Target_Update_Freq = 10 # update target network every 10 episodes

epsilon = 1.0
learning_rate = 0.01
gamma = 0.99
episodes = 1000

for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False

    max_steps = 200
    steps = 0
    while not done and steps < max_steps:
        # steps < max_steps done so that agent doesn't wander around aimlessly
        steps += 1
        # epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = np.random.randint(4)
        else:
            q_values = nn.forward(state)
            action = np.argmax(q_values)

        # taking action
        next_state_env, reward, done = env.step(action)
        next_state = env.get_state()
        total_reward += reward 

        # Store transition in replay buffer
        buffer.push(state, action, reward, next_state, done)

        # Training Step
        if buffer.is_ready(batch_size):
            batch = buffer.sample(batch_size)
            
            # Unpack the batch (numpy arrays across batch dimension)
            # states: (2, batch_size), actions: (batch_size,), etc.
            b_states = np.hstack([t[0] for t in batch]) 
            b_actions = np.array([t[1] for t in batch])
            b_rewards = np.array([t[2] for t in batch])
            b_next_states = np.hstack([t[3] for t in batch])
            b_dones = np.array([t[4] for t in batch])
            
            # Predict next Q-values using the TARGET NETWORK
            next_q = target_nn.forward(b_next_states)
            max_next_q = np.max(next_q, axis=0) # shape (batch_size,)
            
            # Calculate targets for the current state using MAIN NETWORK
            current_q_pred = nn.forward(b_states)
            target = current_q_pred.copy()
            
            # Apply Bellman equation to the specific actions taken
            for i in range(batch_size):
                target[b_actions[i], i] = b_rewards[i] + gamma * max_next_q[i] * (not b_dones[i])
                
            # Perform backward pass and update weights averaged across batch
            # Notice nn.forward(b_states) ran last on `nn`, so internal state self.x and self.a1 are correct for backward pass!
            nn.backward(target)
            nn.update(learning_rate / batch_size)

        state = next_state

    # Sync Target Network
    if episode % Target_Update_Freq == 0:
        target_nn.copy_weights(nn)

    # decaying epsilon / increasing greed
    epsilon = max(0.1, epsilon * 0.995)

    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")