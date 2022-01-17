import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
import random
from collections import deque

def get_random_states(env, target_obs_size, num_states, num_actions):
    states = []
    while len(states) < num_states:
        observations_q = initialize_obs_q(target_obs_size)
        observation = env.reset()
        done = False
        while not done:    
            action = random.randint(0, num_actions-1)
            observation = preprocess(observation, target_obs_size)
            observations_q.append(observation)
            if random.random() < 0.5:
                state = concate_observations(observations_q)
                states.append(state)
            observation, _, done, _ = env.step(action)
    return states[:num_states]

def initialize_obs_q(target_obs_size):
    return deque(
        [
            torch.zeros(target_obs_size),
            torch.zeros(target_obs_size),
            torch.zeros(target_obs_size),
            torch.zeros(target_obs_size)
        ],
        maxlen=4
    )

def concate_observations(q):
    # stacked_observations = [q[x].permute(1,2,0) for x in range(len(q))]
    # plot(stacked_observations)
    # state = torch.cat([q[x] for x in range(len(q))])
    state = torch.unsqueeze(torch.cat([q[x] for x in range(len(q))]), 0)
    return state

def plot(imgs, row_title=None, **imshow_kwargs):
    imgs = [imgs]
    num_rows = len(imgs)
    num_cols = len(imgs[0])
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), cmap='gray', **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()

def preprocess(observation, target_size):
    observation = torch.Tensor(observation).permute(2, 0, 1)
    observation = F.rgb_to_grayscale(observation)
    observation = F.resize(observation, size=(target_size[1], target_size[2]))
    return observation

class LinearScheduler:
    def __init__(self, initial_eps, final_eps, total_steps):
        self.initial_eps = initial_eps
        self.final_eps = final_eps
        self.total_steps = total_steps

    def value(self, step):
        fraction = min(float(step)/self.total_steps, 1.0)
        return self.initial_eps + fraction * (self.final_eps - self.initial_eps)

class ReplayMemory:
    def __init__(self, max_size):
        self.buffer = [None] * max_size
        self.max_size = max_size
        self.index = 0
        self.size = 0

    def append(self, obj):
        old_obj = self.buffer[self.index]
        del old_obj
        self.buffer[self.index] = obj
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size):
        indices = random.sample(range(self.size), batch_size)
        return [self.buffer[index] for index in indices]