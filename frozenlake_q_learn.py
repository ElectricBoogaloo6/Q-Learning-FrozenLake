#### Frozen Lake - https://www.gymlibrary.dev/environments/toy_text/frozen_lake/
#### Action Space = Discrete(4) | Observation Space = Discrete(16) 

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import random

from utils import one_hot_encode

env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=False) # render_mode="human"

# Obs and action spaces: 
obs_space = env.observation_space
act_space = env.action_space
print(f"Observation spaces: {obs_space}")
print(f"Action spaces: {act_space}")


# Parameters
max_episodes = 20000
max_moves = 250

epsilon = 1.0 # to encourage exploration early
epsilon_min = 0.01 
epsilon_decay = 0.995 # to decay per episode

gamma = 0.99 # discount

learning_rate = 0.001

# live updating
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [])
ax.set_xlim(0, max_episodes)
ax.set_ylim(0, max_moves)
ax.set_xlabel("Episode")
ax.set_ylabel("Total Reward")

# Neural Net for Q-table
class QNetwork(nn.Module):
    def __init__(self, state_size=64, action_size=4):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 164)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(164, action_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
q_network = QNetwork().to(device)

# Optimizer
optimizer = optim.Adam(q_network.parameters(), learning_rate)
    
# Initial reset
obs, _ = env.reset()

cum_success = 0
history = []

# Training loop
for episode in range(max_episodes):
    # print(f"in episode: {episode}")
    obs, _ = env.reset()
    done = False
    total_reward = 0
    curr_move = 0

    while not done and curr_move < max_moves:

        state = one_hot_encode(obs, 64)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            with torch.no_grad():
                q_values = q_network(state_tensor)
            action = torch.argmax(q_values).item()  # Exploit

        next_obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        next_state = one_hot_encode(next_obs, 64)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

        next_q_values = q_network(next_state_tensor)
        max_next_q = torch.max(next_q_values).item()
        if done == True:
            target_q = reward
        else:
            target_q = reward + (gamma * max_next_q)
        target_q = torch.tensor([target_q], dtype=torch.float32,device=device)
        q_value = q_network(state_tensor)[0][action]
        # print(f"Current Q Value: {q_value} in state: {obs} with action {action}")


        # loss and network update
        loss = (q_value - target_q) ** 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print(f"Action taken: {action}, Reward: {reward}, Next State: {next_obs}")
        obs = next_obs
        curr_move += 1

    cum_success += total_reward
    cum_rate = cum_success / (episode + 1)
    history.append(cum_rate)

    if episode % 100 == 0:
        print(f"in episode: {episode}")
        line.set_data(range(len(history)), history)
        ax.set_ylim(0, 1)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    continue

env.close()
plt.ioff()
plt.show()
torch.save(q_network.state_dict(), 'q_network_NN_frozenlake_8x8.pth')

# to load the model: 
# q_network = QNetwork()
# q_network.load_state_dict(torch.load('q_network_frozenlake.pth'))
# q_network.eval()