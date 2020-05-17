import numpy as np

from gw_collect import Gridworld
import pygame as pg
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from vicero.algorithms.deepqlearning import DQN
from multitask_env import MultitaskEnvironment
import random

def plot(history):
    plt.figure(2)
    plt.clf()
    durations_t = torch.DoubleTensor(history)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy(), c='lightgray', linewidth=1)

    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), c='green')
            
    plt.pause(0.001)

class LinRegNet(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(LinRegNet, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 32)
        self.fc2 = nn.Linear(32, n_outputs)
        self.linreg = nn.Linear(n_inputs, n_outputs)

    def forward(self, x):
        x = torch.flatten(x)
        #return self.linreg(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

env_list = []

for i in range(10):
    env_list.append(Gridworld(width=4, height=4, cell_size=32, seed=i))

env = MultitaskEnvironment(env_list)

dqn = DQN(env, qnet=LinRegNet(64, 4).double(), plotter=plot, render=False, memory_length=2000, gamma=.99, alpha=.001, epsilon_start=0.1, plot_durations=True)
dqn.train(500, 4, plot=True)

for seed in env.env_scores:
    print('seed={}, score={}'.format(seed, np.mean(env.env_scores[seed])))