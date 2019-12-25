from gridworld import Gridworld
import pygame as pg

scale = 32
env = Gridworld(scale, width=8, height=8)

pg.init()
screen = pg.display.set_mode((scale * len(env.board[0]), scale * len(env.board)))
env.screen = screen

while True:
    env.step(env.action_space.sample())
    env.render()