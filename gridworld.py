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
    'box_edge' : (60, 40, 0),
    'goal' : (10, 10, 10),
    'goal_edge' : (20, 20, 20)
}

class Gridworld(gym.Env):
    UP, DOWN, LEFT, RIGHT = range(4)

    def __init__(self, scale, width=8, height=16):
        self.action_space = spaces.Discrete(4)
        board = np.zeros((height, width))
        self.width = width
        self.height = height
        
        self.time = 0
        self.cutoff = 5000

        self.board = np.array(board)
        self.board_interactive = np.array(board)
        self.board_goal = np.array(board)
        
        # Wall
        #for i in range(1, width - 1):
        #    self.board[2][i] = 1

        self.size = len(board)
        self.cell_size = scale
        self.screen = None

        self.agent_pos = (np.random.randint(0, self.width), np.random.randint(0, self.height))
        while self.board[self.agent_pos[1]][self.agent_pos[0]] == 1:
            self.agent_pos = (np.random.randint(0, self.width), np.random.randint(0, self.height))
        
        box_pos = (np.random.randint(0, self.width), np.random.randint(0, self.height))
        while self.board[box_pos[1]][box_pos[0]] == 1 or box_pos == self.agent_pos:
            box_pos = (np.random.randint(0, self.width), np.random.randint(0, self.height))
        
        self.board_interactive[box_pos[1]][box_pos[0]] = 1
        
        goal_pos = (np.random.randint(0, self.width), np.random.randint(0, self.height))
        while self.board[goal_pos[1]][goal_pos[0]] == 1 or goal_pos == self.agent_pos or goal_pos == box_pos:
            goal_pos = (np.random.randint(0, self.width), np.random.randint(0, self.height))
        
        self.board_goal[goal_pos[1]][goal_pos[0]] = 1
        
    def reset(self):
        self.time = 0
        board = np.zeros((self.height, self.width))
        self.board = np.array(board)
        self.board_interactive = np.array(board)
        self.board_goal = np.array(board)
        
        #for i in range(1, self.width - 1):
        #    self.board[2][i] = 1
        
        self.agent_pos = (np.random.randint(0, self.width), np.random.randint(0, self.height))
        while self.board[self.agent_pos[1]][self.agent_pos[0]] == 1:
            self.agent_pos = (np.random.randint(0, self.width), np.random.randint(0, self.height))
        
        box_pos = (np.random.randint(0, self.width), np.random.randint(0, self.height))
        while self.board[box_pos[1]][box_pos[0]] == 1 or box_pos == self.agent_pos:
            box_pos = (np.random.randint(0, self.width), np.random.randint(0, self.height))
        
        self.board_interactive[box_pos[1]][box_pos[0]] = 1
        
        goal_pos = (np.random.randint(0, self.width), np.random.randint(0, self.height))
        while self.board[goal_pos[1]][goal_pos[0]] == 1 or goal_pos == self.agent_pos or goal_pos == box_pos:
            goal_pos = (np.random.randint(0, self.width), np.random.randint(0, self.height))
        
        self.board_goal[goal_pos[1]][goal_pos[0]] = 1
        

        agent_pos_matrix = np.zeros((self.height, self.width))
        agent_pos_matrix[self.agent_pos[1]][self.agent_pos[0]] = 1

        state = np.array([[
            self.board,
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
                    if self.board[box_target_y][box_target_x] == 0 and \
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

                        
            elif self.board[target_y][target_x] == 0:
                self.agent_pos = (target_x, target_y)

        agent_pos_matrix = np.zeros((self.height, self.width))
        agent_pos_matrix[self.agent_pos[1]][self.agent_pos[0]] = 1

        state = np.array([[
            self.board,
            self.board_interactive,
            self.board_goal,
            agent_pos_matrix
        ]])

        return state, reward, done, {}
    
    def resolve_corner(self, target_x, target_y, box_target_x, box_target_y):
        diff_x = target_x - box_target_x 
        diff_y = target_y - box_target_y
        
        if diff_x == 0: # vertical
            left = target_x - 1 not in range(self.width) or \
                self.board[target_y][target_x - 1] == 1 or \
                self.board_interactive[target_y][target_x - 1] == 1
            right = target_x + 1 not in range(self.width) or \
                self.board[target_y][target_x + 1] == 1 or \
                self.board_interactive[target_y][target_x + 1] == 1
            if left:
                return True, target_x, target_y, target_x + 1, target_y
            if right:
                return True, target_x, target_y, target_x - 1, target_y
            return True, target_x, target_y, target_x, target_y + diff_y
                
        if diff_y == 0: # horizontal
            up = target_y - 1 not in range(self.height) or \
                self.board[target_y - 1][target_x] == 1 or \
                self.board_interactive[target_y - 1][target_x] == 1
            down = target_y + 1 not in range(self.height) or \
                self.board[target_y + 1][target_x] == 1 or \
                self.board_interactive[target_y + 1][target_x] == 1
            if up:
                return True, target_x, target_y, target_x, target_y + 1
            if down:
                return True, target_x, target_y, target_x, target_y - 1
            return True, target_x, target_y, target_x + diff_x, target_y
            
        return False, target_x, target_y, box_target_x, box_target_y
                

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
                elif self.board_goal[j][i] == 1: 
                    pg.draw.rect(screen, colors['goal'], cell)
                    pg.draw.rect(screen, colors['goal_edge'], cell, 1)
                else:
                    pg.draw.rect(screen, colors['floor'], cell)
                    pg.draw.rect(screen, colors['floor_edge'], cell, 1)
        
        cell = pg.Rect(self.cell_size * self.agent_pos[0], self.cell_size * self.agent_pos[1], self.cell_size, self.cell_size)
        
        pg.draw.rect(screen, (100, 0, 0), cell)
        pg.draw.rect(screen, (90, 0, 0), cell, 1)
        

    
    def render(self, mode=''):
        self.draw(self.screen)    
        pg.display.flip()