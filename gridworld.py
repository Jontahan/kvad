import numpy as np
import pygame as pg
import gym
from gym import spaces

colors = {
    'floor' : (80, 80, 80),
    'floor_edge' : (60, 60, 60),
    'wall' : (50, 50, 50),
    'wall_edge' : (30, 30, 30),
    'box' : (80, 60, 0),
    'box_edge' : (60, 40, 0)
}

class Gridworld(gym.Env):
    UP, DOWN, LEFT, RIGHT = range(4)

    def __init__(self, scale, width=8, height=16):
        self.action_space = spaces.Discrete(4)
        board = np.zeros((height, width))
        self.width = width
        self.height = height
        
        self.board = np.array(board)
        self.board_interactive = np.array(board)
        self.board_interactive[6][5] = 1
        for i in range(1, width - 1):
            self.board[4][i] = 1
        
        self.size = len(board)
        self.cell_size = scale
        self.screen = None
        self.agent_pos = (0, 0)
    
    def reset(self):
        board = np.zeros((self.height, self.width))
        self.board = np.array(board)
        self.falling_piece_pos = (np.random.randint(0, self.width - 3), 0)
        self.falling_piece_shape = self.piece_types[np.random.randint(0, len(self.piece_types))]
        self.subframe = 0

        piece = np.array(np.zeros((self.height, self.width)))
        for i in range(4):
            for j in range(4):
                if self.falling_piece_shape[j][i] == 1:
                    pos = (i + self.falling_piece_pos[0], j + self.falling_piece_pos[1])
                    piece[pos[1]][pos[0]] = 1
        self.time = 0

        
        timing_layer = np.zeros((self.height, self.width))
        
        state = np.array([[
            self.board,
            piece,
            timing_layer
        ]])

        return state
        
    def resolve_lines(self):
        removed = 0
        for i in range(len(self.board)):
            line = self.board[i]
            if all(x == 1 for x in line):
                removed = removed + 1
                for j in range(i - 1):
                    self.board[i - j] = self.board[i - j - 1]
        return removed
        
    def step(self, action):
        done = False
        reward = 0
        
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
                    if self.board[box_target_y][box_target_x] == 0 and \
                       self.board_interactive[box_target_y][box_target_x] == 0:
                        self.agent_pos = (target_x, target_y)
                        self.board_interactive[target_y][target_x] = 0
                        self.board_interactive[box_target_y][box_target_x] = 1
            elif self.board[target_y][target_x] == 0:
                self.agent_pos = (target_x, target_y)

        agent_pos_matrix = np.zeros((self.height, self.width))

        state = [
            self.board,
            self.board_interactive,
            agent_pos_matrix
        ]

        return state, reward, done, {}
    
    def draw(self, screen, heatmap=None):
        for i in range(len(self.board[0])):
            for j in range(len(self.board)):
                cell = pg.Rect(self.cell_size * i, self.cell_size * j, self.cell_size, self.cell_size)
                
                if self.board[j][i] == 1: 
                    pg.draw.rect(screen, colors['wall'], cell)
                    pg.draw.rect(screen, colors['wall_edge'], cell, 1)
                elif self.board_interactive[j][i] == 1: 
                    pg.draw.rect(screen, colors['box'], cell)
                    pg.draw.rect(screen, colors['box_edge'], cell, 1)
                else:
                    pg.draw.rect(screen, colors['floor'], cell)
                    pg.draw.rect(screen, colors['floor_edge'], cell, 1)
        
        cell = pg.Rect(self.cell_size * self.agent_pos[0], self.cell_size * self.agent_pos[1], self.cell_size, self.cell_size)
        
        pg.draw.rect(screen, (100, 0, 0), cell)
        pg.draw.rect(screen, (90, 0, 0), cell, 1)
        

    
    def render(self, mode=''):
        self.draw(self.screen)    
        pg.display.flip()