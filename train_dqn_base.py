from gridworld_base import Gridworld
import pygame as pg
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from vicero.algorithms.deepqlearning import DQN

scale = 24
env = Gridworld(scale, width=6, height=6)

pg.init()
screen = pg.display.set_mode((scale * len(env.board[0]), scale * len(env.board)))
env.screen = screen
clock = pg.time.Clock()

"""
while True:
    env.step(env.action_space.sample())
    env.render()
"""

def plot(history):
        plt.figure(2)
        plt.clf()
        durations_t = torch.DoubleTensor(history)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy(), c='lightgray', linewidth=1)

        his = 50
        if len(durations_t) >= his:
            means = durations_t.unfold(0, his, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(his - 1), means))
            plt.plot(means.numpy(), c='green')
            print(np.std(durations_t[-his:].tolist()))
            
        plt.pause(0.001)

"""
class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        
        self.conv = nn.Conv2d(4, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(144, 32)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 5)
        
    def forward(self, x):
        x = F.relu(self.conv(x))
        #x = self.pool(x)
        #x = F.relu(self.conv2(x))
        x = torch.flatten(x)
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
"""

class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(256, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 4)
        
    def forward(self, x):
        x = torch.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

dqn = DQN(env, qnet=PolicyNet().double(), plotter=plot, render=True, memory_length=2000, gamma=.99, alpha=.001, epsilon_start=0.1)
dqn.train(2000, 4, plot=True, verbose=True)