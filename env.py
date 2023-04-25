# Implemnetation of tetris and hamin .py are not REALLY reinforcement learning environments
# Take a look at this sample code:

"""
import gym
env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)
for _ in range(1000):
   action = policy(observation)  # User-defined policy function
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()
env.close()
"""

# So in other words, this .py file will attempt to reconstruct tetris but in this format, using the tetris classes in tetris.py

from tetris import Tetris, WHITE, GRAY, BLACK, colors
import pygame
import numpy as np
from copy import deepcopy

class TetrisEnv:
    def __init__(self, height, width, render=False) -> None:
        self.h = height
        self.w = width
        self.rend = render

        # if render, initialize some stuff
        if render:
            pygame.init()

            # Define some colors
            self.BLACK = (0, 0, 0)
            self.WHITE = (255, 255, 255)
            self.GRAY = (128, 128, 128)

            size = (400, 500)
            self.screen = pygame.display.set_mode(size)

            pygame.display.set_caption("Tetris")

            # Loop until the user clicks the close button.
            self.done = False
            self.clock = pygame.time.Clock()
            self.fps = 25

    def _gen_ob (self): # generate observation
        # set everything to 1 if obstacle
        ob = deepcopy(self.game.field)
        for i in range(self.h):
            for j in range(self.w):
                if ob[i][j] > 0:
                    ob[i][j] = 0.5

        # create the FIGURE
        for i in range(4):
            for j in range(4):
                p = i * 4 + j
                if p in self.game.figure.image():
                    ob[i + self.game.figure.y][j + self.game.figure.x] = 2

        nextFig = np.zeros((4,4))
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.game.figure.image():
                    nextFig[i][j] = 1

        a = np.array(ob).flatten()
        return np.concatenate((a, nextFig.flatten()), axis=0)
    
    def reset (self):
        self.game = Tetris(self.h, self.w)
        return self._gen_ob()


    def step (self, action):
        prevScore = self.game.score 
        
        if action == 0: self.game.rotate(reverse=False)
        elif action == 1: self.game.rotate(reverse=True)
        elif action == 2: self.game.go_side(-1)
        elif action == 3: self.game.go_side(1)
        elif action == 4: self.game.go_space()
        # there's an another action that just does nothing, action = 5
        self.game.go_down()

        reward = self.game.score - prevScore
        is_terminated = False
        if self.game.state == "gameover":
            reward = -25 
            is_terminated = True

        if self.game.figure is None:
            self.game.new_figure()

        return self._gen_ob(), reward, is_terminated


    def render(self):
        if self.rend:

            self.screen.fill(WHITE)

            for i in range(self.game.height):
                for j in range(self.game.width):
                    pygame.draw.rect(self.screen, GRAY, 
                        [self.game.x + self.game.zoom * j, 
                         self.game.y + self.game.zoom * i, 
                         self.game.zoom, self.game.zoom]
                    , 1)

                    if self.game.field[i][j] > 0:
                        pygame.draw.rect(
                            self.screen, 
                            colors[self.game.field[i][j]],
                            [self.game.x + self.game.zoom * j + 1, self.game.y + self.game.zoom * i + 1, self.game.zoom - 2, self.game.zoom - 1]
                        )

            if self.game.figure is not None:
                for i in range(4):
                    for j in range(4):
                        p = i * 4 + j
                        if p in self.game.figure.image():
                            pygame.draw.rect(self.screen, colors[self.game.figure.color],
                                [self.game.x + self.game.zoom * (j + self.game.figure.x) + 1,
                                self.game.y + self.game.zoom * (i + self.game.figure.y) + 1,
                                self.game.zoom - 2, self.game.zoom - 2]
                            )

            font = pygame.font.SysFont('Calibri', 25, True, False)
            font1 = pygame.font.SysFont('Calibri', 65, True, False)
            text = font.render("Score: " + str(self.game.score), True, BLACK)
            text_game_over = font1.render("Game Over", True, (255, 125, 0))
            text_game_over1 = font1.render("Press ESC", True, (255, 215, 0))

            self.screen.blit(text, [0, 0])
            if self.game.state == "gameover":
                self.screen.blit(text_game_over, [20, 200])
                self.screen.blit(text_game_over1, [25, 265])

            pygame.display.flip()
            self.clock.tick(self.fps)