import numpy as np
import pygame as pg
import gym
from gym import spaces

# For visualization
colors = {
    'floor' : (80, 80, 80),
    'floor_edge' : (70, 70, 70),
    'wall' : (50, 50, 50),
    'wall_edge' : (40, 40, 40),
    'box' : (80, 60, 0),
    'box_edge' : (60, 40, 0),
    'goal' : (10, 10, 10),
    'goal_edge' : (20, 20, 20)
}

class Gridworld(gym.Env):
    # Action space
    UP, DOWN, LEFT, RIGHT = range(4)

    def __init__(self, cell_size=16, width=8, height=8, walls=None, screen=None, agent_pos=(0, 0), goal_pos=(0, 0)):
        self.action_space = spaces.Discrete(4)
        
        # config
        self.init_agent_pos = agent_pos
        self.goal_pos = goal_pos

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
        self.board_goal = np.zeros((self.height, self.width))
        self.agent_pos = self.init_agent_pos
        self.board_goal[self.goal_pos[1]][self.goal_pos[0]] = 1
        
        agent_pos_matrix = np.zeros((self.height, self.width))
        agent_pos_matrix[self.agent_pos[1]][self.agent_pos[0]] = 1

        state = np.array([[
            self.board_walls,
            self.board_interactive,
            self.board_goal,
            agent_pos_matrix
        ]])

        return state
        
    def step(self, action):
        done = self.time > self.cutoff
        self.time += 1
        reward = -1
        
        target_x, target_y = self.agent_pos

        if action == Gridworld.UP:    target_y -= 1
        if action == Gridworld.DOWN:  target_y += 1
        if action == Gridworld.LEFT:  target_x -= 1
        if action == Gridworld.RIGHT: target_x += 1
        
        if target_x in range(self.width) and target_y in range(self.height):
            if self.board_interactive[target_y][target_x] == 1:
                box_target_x = target_x
                box_target_y = target_y

                if action == Gridworld.UP:    box_target_y -= 1
                if action == Gridworld.DOWN:  box_target_y += 1
                if action == Gridworld.LEFT:  box_target_x -= 1
                if action == Gridworld.RIGHT: box_target_x += 1
                
                if box_target_x in range(self.width) and box_target_y in range(self.height):
                    if self.board_walls[box_target_y][box_target_x] == 0 and \
                       self.board_interactive[box_target_y][box_target_x] == 0:
                        self.agent_pos = (target_x, target_y)
                        self.board_interactive[target_y][target_x] = 0
                        self.board_interactive[box_target_y][box_target_x] = 1
                else:
                    move, target_x, target_y, box_target_x, box_target_y = self.resolve_corner(target_x, target_y, box_target_x, box_target_y)
                    if move:
                        self.agent_pos = (target_x, target_y)
                        self.board_interactive[target_y][target_x] = 0
                        self.board_interactive[box_target_y][box_target_x] = 1
                
                if box_target_x in range(self.width) and box_target_y in range(self.height) and self.board_goal[box_target_y][box_target_x] == 1:
                    done = True
                    reward = 10

            if target_x in range(self.width) and target_y in range(self.height) and self.board_goal[target_y][target_x] == 1:
                    done = True
                    reward = 10

                        
            elif self.board_walls[target_y][target_x] == 0:
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

        return state, reward, done, { 'agent_pos_i' : agent_pos_i }
    
    def resolve_corner(self, target_x, target_y, box_target_x, box_target_y):
        diff_x = target_x - box_target_x 
        diff_y = target_y - box_target_y
        
        if diff_x == 0: # vertical
            left = target_x - 1 not in range(self.width) or \
                self.board_walls[target_y][target_x - 1] == 1 or \
                self.board_interactive[target_y][target_x - 1] == 1
            right = target_x + 1 not in range(self.width) or \
                self.board_walls[target_y][target_x + 1] == 1 or \
                self.board_interactive[target_y][target_x + 1] == 1
            if left:
                return True, target_x, target_y, target_x + 1, target_y
            if right:
                return True, target_x, target_y, target_x - 1, target_y
            return True, target_x, target_y, target_x, target_y + diff_y
                
        if diff_y == 0: # horizontal
            up = target_y - 1 not in range(self.height) or \
                self.board_walls[target_y - 1][target_x] == 1 or \
                self.board_interactive[target_y - 1][target_x] == 1
            down = target_y + 1 not in range(self.height) or \
                self.board_walls[target_y + 1][target_x] == 1 or \
                self.board_interactive[target_y + 1][target_x] == 1
            if up:
                return True, target_x, target_y, target_x, target_y + 1
            if down:
                return True, target_x, target_y, target_x, target_y - 1
            return True, target_x, target_y, target_x + diff_x, target_y
            
        return False, target_x, target_y, box_target_x, box_target_y
                

    def draw(self, screen, heatmap=None):
        for i in range(len(self.board_walls[0])):
            for j in range(len(self.board_walls)):
                cell = pg.Rect(self.cell_size * i, self.cell_size * j, self.cell_size, self.cell_size)
                cell_border = pg.Rect(self.cell_size * i, self.cell_size * j, self.cell_size + 1, self.cell_size + 1)
                
                if self.board_walls[j][i] == 1: 
                    pg.draw.rect(screen, colors['wall'], cell)
                    pg.draw.rect(screen, colors['wall_edge'], cell_border, 1)
                elif self.board_interactive[j][i] == 1: 
                    pg.draw.rect(screen, colors['box'], cell)
                    pg.draw.rect(screen, colors['box_edge'], cell_border, 1)
                elif self.board_goal[j][i] == 1: 
                    pg.draw.rect(screen, colors['goal'], cell)
                    pg.draw.rect(screen, colors['goal_edge'], cell_border, 1)
                else:
                    pg.draw.rect(screen, colors['floor'], cell)
                    pg.draw.rect(screen, colors['floor_edge'], cell_border, 1)
        
        agent_cell = pg.Rect(self.cell_size * self.agent_pos[0] + 2, self.cell_size * self.agent_pos[1] + 2, self.cell_size - 4, self.cell_size - 4)
        
        pg.draw.rect(screen, (100, 0, 0), agent_cell)
        pg.draw.rect(screen, (90, 0, 0), agent_cell, 1)
        

    
    def render(self, mode=''):
        self.draw(self.screen)    
        pg.display.flip()