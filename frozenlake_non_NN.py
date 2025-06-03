import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import random


map_name = "8x8"
is_slippery = False
name = 'FrozenLake-v1'

env = gym.make(name, desc=None, map_name=map_name, is_slippery=is_slippery) # render_mode="human"

# Obs and action spaces: 
obs_space = env.observation_space
act_space = env.action_space
print(f"Observation spaces: {obs_space}")
# 64
print(f"Action spaces: {act_space}")
# 4

# Params
max_episodes = 40000
max_moves = 250

epsilon = 1.0 # to encourage exploration early
epsilon_min = 0.01 
epsilon_decay = 0.995 # to decay per episode

gamma = 0.99 # discount
alpha = 0.5


# live updating
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [])
ax.set_xlim(0, max_episodes)
ax.set_ylim(0, max_moves)
ax.set_xlabel("Episode")
ax.set_ylabel("Total Reward")

q_table = np.zeros((64, 4))

# Initial reset
obs, _ = env.reset()
cum_success = 0
history = []

for episode in range(max_episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    curr_move = 0

    while not done and curr_move < max_moves:
        state = obs
        
        if np.random.rand() < epsilon:
            action = np.random.randint(0,4)
        else:
            action = np.argmax(q_table[state]).item()

        next_obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        next_state = next_obs

        if done == True:
            q_table[state][action] = q_table[state][action] + alpha * (reward - q_table[state][action])
        else:
            q_table[state][action] = q_table[state][action] + alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])
        

        obs = next_obs
        curr_move += 1

    if episode % 100 == 0:
        print(f"Q-table at update: {q_table[state]}")
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

np.save(f'q_table_{name}_{map_name}_not_slippery.npy', q_table)
env.close()
plt.savefig(f"{name}_{map_name}_not_slippery_graph_q-table.png")
plt.ioff()
plt.show()
