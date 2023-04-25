from env import * 
from time import sleep
from random import randint
from AI import *
import os

# Hyperparams
w = 10
h = 20

# Actual vars
env = TetrisEnv(height=h, width=w, render=True)
ob = env.reset()
num_epi_trained = 1_000_000
agent = Agent(
    h,
    w,
    1,
    1,
    1,
    1,
    1,
    "./model.pth"
)

type = 2
# 0 == agent 
# 1 == random move
# 2 == player

for i in range(num_epi_trained):
    terminated = False

    rew = 0
    numTimesteps = 0
    while not terminated:
        if type == 0:
            action = agent.act(ob)
        elif type == 1:
            action = randint(0, 5)
        elif type == 2:
            action = 5
            for event in list(pygame.event.get()):
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        action = 2
                    if event.key == pygame.K_RIGHT:
                        action = 3
                    if event.key == pygame.K_UP:
                        action = 1
                    if event.key == pygame.K_DOWN: 
                        action = 0
                    if event.key == pygame.K_SPACE:
                        action = 4
                break
        
        nextOb, reward, terminated = env.step(action)
        rew += reward    
        numTimesteps += 1

        ob = nextOb

        env.render()

    st = f"Epi #{i+1} Reward: {(rew / numTimesteps):.3f}"
    wh = " " * (os.get_terminal_size().columns - len(st) - 2)
    print(st)
    
    ob = env.reset()

