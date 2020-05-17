import numpy as np

from gw_collect import Gridworld
import pygame as pg
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from vicero.algorithms.deepqlearning import DQN

"""
walls = [
    [
        [ 0, 0, 0, 0, 0, 0 ],
        [ 0, 0, 1, 0, 0, 0 ],
        [ 0, 1, 1, 1, 1, 0 ],
        [ 0, 0, 1, 0, 0, 0 ],
        [ 0, 0, 0, 0, 0, 1 ],
        [ 0, 0, 1, 0, 0, 0 ]
    ],
    [
        [ 0, 0, 0, 0, 0, 0 ],
        [ 0, 0, 1, 0, 0, 0 ],
        [ 0, 1, 1, 1, 1, 0 ],
        [ 0, 0, 0, 0, 0, 0 ],
        [ 0, 0, 1, 0, 0, 1 ],
        [ 0, 0, 1, 0, 0, 0 ]
    ]
]

env = Gridworld(width=6, height=6, walls=walls, cell_size=32, agent_pos=(0, 4), goal_pos=(4, 4))
"""
"""
walls = [
    [
        [ 0, 0, 0, 0 ],
        [ 0, 1, 1, 0 ],
        [ 0, 0, 1, 0 ],
        [ 0, 0, 1, 0 ]
    ],
    [
        [ 0, 0, 0, 0 ],
        [ 0, 0, 1, 0 ],
        [ 0, 0, 1, 0 ],
        [ 0, 0, 1, 0 ]
    ]
]
"""

env = Gridworld(width=6, height=6, cell_size=32, agent_pos=(0, 3), food_pos=[(0, 0), (3, 3), (4, 5), (2, 0)])

pg.init()
screen = pg.display.set_mode((env.cell_size * env.width, env.cell_size * env.height))
env.screen = screen
clock = pg.time.Clock()

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
            
        plt.pause(0.001)

class DenseNet(nn.Module):
    def __init__(self, input_size, layer_sizes):
        super(DenseNet, self).__init__()
        self.layers = [nn.Linear(input_size, layer_sizes[0])]

        for i in range(1, len(layer_sizes)):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
        
        self.nn = nn.Sequential(*self.layers)

    def forward(self, x):
        x = torch.flatten(x)
        return self.nn.forward(x)

gamma = .95
alpha = .002

num_episodes = 2000
convergence_durations = []
for i in range(50):
    print('Simulation {}/{}'.format(i, 50))
    net = DenseNet(144, [16, 4]).double()
    dqn = DQN(env, qnet=net, plotter=plot, render=True, memory_length=2000, gamma=gamma, alpha=alpha, epsilon_start=0.3, caching_interval=3000)

    for e in range(num_episodes):
        dqn.train_episode(e, num_episodes, 16, plot=True, verbose=False)
        if e > 50 and np.std(dqn.history[-50:]) < 20:
            print('Early stop after {} iterations'.format(e))
            convergence_durations.append(e)
            break

    if len(convergence_durations) <= i:
        convergence_durations.append(env.cutoff)

print('mean duration: {}'.format(np.mean(convergence_durations)))