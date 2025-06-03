import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import random

from utils import record_video


REC_VID = True

map_name = "4x4"
is_slippery = False
name = 'FrozenLake-v1'

env = gym.make(name, desc=None, map_name=map_name, is_slippery=is_slippery, render_mode="rgb_array") # render_mode="human"
q_table = np.load(f'q_table_{name}_{map_name}_not_slippery.npy')

# params
n_eval_episodes = 100
max_moves = 20

cum_success = 0
history = []

if REC_VID == True:
    record_video(env, q_table, f"{name}_{map_name}_non_slippery.gif", using_NN=False)
else:
    for episode in range(n_eval_episodes):
        obs, _ = env.reset()
        curr_move = 0
        done = False
        total_reward = 0

        while not done and curr_move < max_moves:
            state = obs
            action = np.argmax(q_table[state]).item()
            next_obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            next_state = next_obs

            obs = next_obs
            curr_move += 1

        cum_success += total_reward
        print(f"Cummulative success: {cum_success}")
        cum_rate = cum_success / (episode + 1)
        print(f"cummulative rate: {cum_rate}")
        history.append(cum_rate)

    env.close()