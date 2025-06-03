import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import random

from utils import one_hot_encode, record_video


REC_VID = True

map_name = "8x8"
is_slippery = True
name = 'FrozenLake-v1'

env = gym.make(name, desc=None, map_name=map_name, is_slippery=is_slippery, render_mode="rgb_array") # render_mode="human"

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

max_episodes = 100
max_moves = 250

q_network = QNetwork()
q_network.load_state_dict(torch.load(f'q_table_{name}_{map_name}_slippery.pth'))
q_network.eval()

obs, _ = env.reset()

cum_success = 0
history = []

if REC_VID == True:
    record_video(env, q_network, f"{name}_{map_name}_slippery.gif", using_NN=True)
else:
    for episode in range(max_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        curr_move = 0

        while not done and curr_move < max_moves:
            state = one_hot_encode(obs, 64)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            with torch.no_grad():
                q_values = q_network(state_tensor)

            action = torch.argmax(q_values).item()
            next_obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            next_state = one_hot_encode(next_obs, 64)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            obs = next_obs
            curr_move += 1

        cum_success += total_reward
        print(f"Cummulative success: {cum_success}")
        cum_rate = cum_success / (episode + 1)
        print(f"cummulative rate: {cum_rate}")
        history.append(cum_rate)

    env.close()