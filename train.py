from env import * 
from time import sleep
from random import randint
from AI import *
import os

# Hyperparams
w = 10
h = 20
lr = 0.001
mem_cap = 50_000
frameReachProb = 10_000_000
targetFreqUpdate = 5_000
batches = 32
modelPath = "./model.pth"
savePerEpi = 100

# Actual vars
env = TetrisEnv(height=h, width=w)
ob = env.reset()
num_epi_trained = 1_000_000
agent = Agent(
    h=h,
    w=w,
    memCap=mem_cap,
    lr=lr,
    frameReachProb=frameReachProb,
    targetFreqUpdate=targetFreqUpdate,
    batches=batches
)

for i in range(num_epi_trained):
    terminated = False

    rew = 0
    numTimesteps = 0
    while not terminated:
        action = agent.act(ob)
        
        nextOb, reward, terminated = env.step(action)
        rew += reward    
        numTimesteps += 1

        agent.replayMem.store(ob, action, reward, nextOb, terminated)

        ob = nextOb

    st = f"Epi #{i+1} Reward: {(rew / numTimesteps):.3f} ep: {agent.get_ep_prob():.3f}"
    wh = " " * (os.get_terminal_size().columns - len(st) - 2)
    print(st,wh, end='\r')
    
    agent.train()
    ob = env.reset()

    # save model every few often
    if i % savePerEpi == 0:
        agent.save(modelPath)

agent.save(modelPath)
print("Model saved")
