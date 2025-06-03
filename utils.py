import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import imageio

def one_hot_encode(state, state_size=64):
    vector = np.zeros(state_size)
    vector[state] = 1
    return vector

def record_video(env, Qtable, out_directory, fps=1, using_NN=True):
    images = []
    done = False

    obs, _ = env.reset()
    
    img = env.render()
    images.append(img)
    
    while not done:
        if using_NN == True:
            state = one_hot_encode(obs, 64)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = Qtable(state_tensor)
            action = torch.argmax(q_values).item()
            next_obs, reward, done, truncated, info = env.step(action)
            next_state = one_hot_encode(next_obs, 64)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            obs = next_obs
        else:
            state = obs
            action = np.argmax(Qtable[state]).item()
            next_obs, reward, done, truncated, info = env.step(action)
            next_state = next_obs
            obs = next_obs
        
        img = env.render()
        images.append(img)
    imageio.mimsave(out_directory, [np.array(img) for i, img in enumerate(images)], fps=fps)