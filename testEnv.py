from env import * 
from time import sleep
from random import randint

env = TetrisEnv(10, 20, render=True)
ob = env.reset()

for _ in range(1000):
    ob, reward, terminated = env.step(randint(0,5))
    if terminated: 
        env.reset()

    env.render()

    sleep(0.01)