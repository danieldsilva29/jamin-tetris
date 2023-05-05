import torch
from torch import nn
from random import randint
from copy import deepcopy

class ReplayMem:
    def __init__(self, capacity) -> None:
        self.capacity = capacity 
        self.mem = []

    def store (self, currentState, action, reward, nextState, isDone):
        self.mem.append([
            currentState,
            action, 
            reward,
            nextState,
            isDone
        ])
        self.mem = self.mem[-self.capacity:]

    def random (self, dev):
        if len(self.mem) > 1:
            randNum = randint(0, len(self.mem)-1)
            m = self.mem[randNum]
        else:
            m = self.mem[0]
        currentState = torch.tensor(m[0]).to(dev).to(torch.float)
        nextState = torch.tensor(m[3]).to(dev).to(torch.float)
        action = m[1]
        reward = m[2]
        isDone = m[4]
        
        return (
            currentState,
            action,
            reward,
            nextState,
            isDone
        )
    
# okay daniel hi


# so the replay memory is basicalyl the "memory" of the AI
# and that will be used for trianing :thumbsup:

# I see
# I am referencing this for now https://github.com/Eshaancoding/DQN/blob/master/Agent.cpp

class Agent (nn.Module):

    def __init__(self, h, w, memCap, lr, frameReachProb, targetFreqUpdate, batches, modelPath=None) -> None:
        super().__init__()
        
        # Actual NN decl
        self.nn = nn.Sequential(
            nn.Linear(h*w+16, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6),  
            nn.Softmax()
        )
        # 6 total actions: rotate, rotate reverse, down, left, right, space

        if modelPath != None:
            self.nn = torch.load(modelPath)
            self.is_testing = True        
        else:
            self.is_testing = False        
            # target nn
            self.target_nn = deepcopy(self.nn)

            # params
            self.replayMem = ReplayMem(memCap)   # replay memory for training / looking at past exp
            self.frProb = frameReachProb         # exploration
            self.tUpdate = targetFreqUpdate      # how much times we want to update the target policy
            self.batches = batches               # the batches i.e the number of samples trained under one policy update

            # Optimizer when training nn
            self.opt = torch.optim.Adam(
                self.nn.parameters(),
                lr=lr
            )

            # loss func
            self.mse = torch.nn.MSELoss()

        self.frames = 0

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

    def get_ep_prob (self):
        prob = 1
        if self.frames <= self.frProb:
            prob = (-0.95 / self.frProb) * self.frames + 1
        else:
            prob = 0.05
        
        return prob

    def act (self, input):
        input = torch.tensor(input).to(self.device).to(torch.float)

        self.nn.eval()
        if not self.is_testing:  # we are training
            self.frames += 1

            israndom = randint(0, 99) < (self.get_ep_prob() * 100)
            if israndom:
                return randint(0, 5)
            else:
                a = self.nn(input.to(self.device)).cpu()
                return torch.argmax(a).item()
        else:
            a = self.nn(input.to(self.device)).cpu()
            return torch.argmax(a).item()
        
    def train (self):
        self.opt.zero_grad()        

        for _ in range(self.batches):
            currentState, action, reward, nextState, isDone = self.replayMem.random(self.device)

            y = reward
            y += isDone * (0.99 * torch.max(self.target_nn(nextState)).cpu().item())

            pred = self.nn(currentState)
            t = pred.detach().clone()
            t[action] = y

            l = self.mse(pred, t) / self.batches # scale loss since we are doing gradient accumulation
            l.backward()

        self.opt.step()

        if self.frames % self.tUpdate == 0:
            self.target_nn.load_state_dict(self.nn.state_dict())

    def save (self, modelPath):
        torch.save(self.nn, modelPath)
