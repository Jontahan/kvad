import numpy as np
import pygame as pg
import gym
from gym import spaces
from itertools import chain, combinations
import random
import os

# For visualization
colors = {
    'floor' : (80, 80, 80),
    'floor_edge' : (70, 70, 70),
    'food' : (2*80, 2*80, 0),
    'food_edge' : (2*70, 2*70, 0),
    'wall' : (50, 50, 50),
    'wall_edge' : (40, 40, 40),
}

pwd = os.path.dirname(os.path.realpath(__file__))

sprites = {
    'agent' : pg.image.load(os.path.join(pwd, 'res', 'tile_agent.png')),
    'gem' : pg.image.load(os.path.join(pwd, 'res', 'tile_gem.png')),
    'wall' : pg.image.load(os.path.join(pwd, 'res', 'tile_wall.png'))
}

class Gridworld(gym.Env):
    # Action space
    UP, DOWN, LEFT, RIGHT = range(4)

    def __init__(self, cell_size=16, width=8, height=8, walls=None, screen=None, agent_pos=(0, 0), food_pos=[(0, 0)], simple_state=False, plot_durations=True, name="Environment", seed=None):
        self.action_space = spaces.Discrete(4)
        self.name = name
        self.seed = seed
        # config
        if seed is None:
            self.init_agent_pos = agent_pos
            self.food_pos = food_pos
        else:
            rng = random.Random(seed)
            self.init_agent_pos = (rng.randint(0, 3), rng.randint(0, 3))
            self.food_pos = []
            n_foods = rng.randint(1, 3)
            for _ in range(n_foods):
                self.food_pos.append((rng.randint(0, 3), rng.randint(0, 3)))
        self.food_pos = list(set(self.food_pos))
        self.food_left = len(self.food_pos)
        self.simple_state = simple_state

        #s = list(self.food_pos)
        #for sset in chain.from_iterable(combinations(s, r) for r in range(len(s)+1)):
        #    print(sset)

        # dimensions
        self.width = width
        self.height = height
        
        # timeout
        self.time = 0
        self.cutoff = 1000

        # wall matrix
        self.walls = walls

        # visualization
        self.cell_size = cell_size
        self.screen = screen

    def reset(self):
        # timeout
        self.time = 0

        # state
        self.board_walls = np.array(self.walls[np.random.randint(0, len(self.walls))]) if self.walls else np.zeros((self.height, self.width))
        self.board_interactive = np.zeros((self.height, self.width))
        for food in self.food_pos:
            self.board_interactive[food[1]][food[0]] = 1
        self.board_goal = np.zeros((self.height, self.width))
        self.agent_pos = self.init_agent_pos
        self.food_left = len(self.food_pos)

        agent_pos_matrix = np.zeros((self.height, self.width))
        agent_pos_matrix[self.agent_pos[1]][self.agent_pos[0]] = 1

        state = np.array([[
            self.board_walls,
            self.board_interactive,
            self.board_goal,
            agent_pos_matrix
        ]])
        
        agent_pos_i = self.width * self.agent_pos[1] + self.agent_pos[0]
        if self.simple_state:
            state = agent_pos_i
        return state
        
    def step(self, action):
        done = self.time > self.cutoff
        self.time += 1
        reward = 0
        
        target_x, target_y = self.agent_pos

        if action == Gridworld.UP:    target_y -= 1
        if action == Gridworld.DOWN:  target_y += 1
        if action == Gridworld.LEFT:  target_x -= 1
        if action == Gridworld.RIGHT: target_x += 1
        
        if target_x in range(self.width) and target_y in range(self.height):
            if self.board_interactive[target_y][target_x] == 1:
                self.board_interactive[target_y][target_x] = 0
                self.food_left -= 1
                reward = 1

            if self.food_left <= 0:
                done = True
            
            if self.board_walls[target_y][target_x] == 0:
                self.agent_pos = (target_x, target_y)

        agent_pos_matrix = np.zeros((self.height, self.width))
        agent_pos_matrix[self.agent_pos[1]][self.agent_pos[0]] = 1
        
        agent_pos_i = self.width * self.agent_pos[1] + self.agent_pos[0]
        
        state = np.array([[
            self.board_walls,
            self.board_interactive,
            self.board_goal,
            agent_pos_matrix
        ]])

        if self.simple_state:
            state = agent_pos_i
        return state, reward, done, { 'agent_pos_i' : agent_pos_i }
    
    def draw(self, screen, heatmap=None):
        for i in range(len(self.board_walls[0])):
            for j in range(len(self.board_walls)):
                cell = pg.Rect(self.cell_size * i, self.cell_size * j, self.cell_size, self.cell_size)
                cell_border = pg.Rect(self.cell_size * i, self.cell_size * j, self.cell_size + 1, self.cell_size + 1)
                
                pg.draw.rect(screen, colors['floor'], cell)
                pg.draw.rect(screen, colors['floor_edge'], cell_border, 1)
        
                if self.board_walls[j][i] == 1: 
                    screen.blit(pg.transform.scale(sprites['wall'], (self.cell_size, self.cell_size)), (self.cell_size * i, self.cell_size * j))
                if self.board_interactive[j][i] == 1: 
                    screen.blit(pg.transform.scale(sprites['gem'], (self.cell_size, self.cell_size)), (self.cell_size * i, self.cell_size * j))
                
        screen.blit(pg.transform.scale(sprites['agent'], (self.cell_size, self.cell_size)), (self.cell_size * self.agent_pos[0], self.cell_size * self.agent_pos[1]))
        
    def get_all_states(self):
        states = []
        
        s = list(self.food_pos)
        food_combinations = []
        for sset in chain.from_iterable(combinations(s, r) for r in range(len(s)+1)):
            food_combinations.append(sset)
        for fcomb in food_combinations:
            for y in range(self.height):
                for x in range(self.width):
                    board_walls = np.zeros((self.height, self.width))
                    board_interactive = np.zeros((self.height, self.width))
                    board_goal = np.zeros((self.height, self.width))
                    
                    agent_pos = (x, y)
                    if agent_pos in fcomb or len(fcomb) < 1:
                        continue

                    for food in fcomb:
                        board_interactive[food[1]][food[0]] = 1

                    agent_pos_matrix = np.zeros((self.height, self.width))
                    agent_pos_matrix[agent_pos[1]][agent_pos[0]] = 1

                    state = np.array([[
                        board_walls,
                        board_interactive,
                        board_goal,
                        agent_pos_matrix
                    ]])

                    states.append(state)
            
        return states
    
    
    def render(self, mode=''):
        self.draw(self.screen)    
        pg.display.flip()