import numpy as np

from gw_collect import Gridworld
import pygame as pg
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from vicero.algorithms.qlearning import Qlearning

#env = Gridworld(width=6, height=6, cell_size=32, agent_pos=(0, 3), food_pos=[(0, 0), (3, 3), (4, 5), (2, 0)])
env = Gridworld(width=4, height=4, cell_size=32, agent_pos=(0, 0), food_pos=[(0, 3), (3, 3)])

pg.init()
screen = pg.display.set_mode((env.cell_size * env.width, env.cell_size * env.height))
env.screen = screen
clock = pg.time.Clock()

def plot(history):
        plt.figure(2)
        plt.clf()
        durations_t = torch.FloatTensor(history)
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

all_mean_diffs = []

all_states = env.get_all_states()        
ql_a = Qlearning(env, n_states=len(all_states), n_actions=env.action_space.n, plotter=plot, epsilon=1.0, epsilon_decay=lambda e, i: e * .998)
ql_b = Qlearning(env, n_states=len(all_states), n_actions=env.action_space.n, plotter=plot, epsilon=1.0, epsilon_decay=lambda e, i: e * .998)
        
for ne in range(0, 10):
    np.random.seed(10)
    num_episodes = 100 #* ne
    convergence_durations = []
    ql_agents = []
    for i in range(2):
        #print('Simulation {}/{}'.format(i, 50))
        
        ql = Qlearning(env, n_states=len(all_states), n_actions=env.action_space.n, plotter=plot, epsilon=1.0, epsilon_decay=lambda e, i: e * .998)
        #dqn = DQN(env, qnet=net, plotter=plot, render=True, memory_length=2000, gamma=gamma, alpha=alpha, epsilon_start=0.3, caching_interval=3000)

        for e in range(num_episodes):
            ql.train(1)
            #dqn.train_episode(e, num_episodes, 16, plot=True, verbose=False)
            #if e > 50 and np.std(ql.history[-50:]) < 1:
            #    print('Early stop after {} iterations'.format(e))
            #    convergence_durations.append(e)
            #    break
        
        ql_agents.append(ql)        

        
    env_diffs = []
    total_visits = []
    for i in range(len(ql_agents[0].Q)):
        total_visits.append(ql_agents[0].state_visits[i] + ql_agents[1].state_visits[i])

    normalized_visits = total_visits / np.linalg.norm(total_visits)
    #normalized_visits = nn.Softmax(dim=-1)(torch.tensor(total_visits))
    #for i in range(len(normalized_visits)):
    #    print('{} -> {}'.format(total_visits[i], normalized_visits[i]))

    for i in range(len(ql_agents[0].Q)):
        for a in range(env.action_space.n):
            env_diffs.append(normalized_visits[i] * abs(ql_agents[0].Q[i][a] - ql_agents[1].Q[i][a]))

        #env_diffs.append(normalized_visits[i] * abs(ql_agents[0].Q[i] - ql_agents[1].Q[i]))

    print('mean difference: {}'.format(np.mean(env_diffs)))
    all_mean_diffs.append(np.mean(env_diffs))

plt.close()
plt.plot(all_mean_diffs)
plt.show()
#print('mean duration: {}'.format(np.mean(convergence_durations)))

# std < 1 for this config