from gridworld import Gridworld
import pygame as pg
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from vicero.algorithms.deepqlearning import DQN

scale = 32
env = Gridworld(scale, width=5, height=5)

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

        his = 100
        if len(durations_t) >= his:
            means = durations_t.unfold(0, his, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(his - 1), means))
            plt.plot(means.numpy(), c='green')
            
        plt.pause(0.001)

class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        
        self.conv = nn.Conv2d(4, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(72, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 5)
        
    def forward(self, x):
        x = F.relu(self.conv(x))
        #x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = torch.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

dqn = DQN(env, qnet=PolicyNet().double(), plotter=plot, render=True, memory_length=20000, gamma=.92, alpha=.0001, epsilon_start=0.5)
dqn.train(1000, 4, plot=True, verbose=True)