import numpy as np

def one_hot_encode(state, state_size=64):
    vector = np.zeros(state_size)
    vector[state] = 1
    return vector

