"""
This class is basically the buffer class, which will be responsible for storing the transitions, 
and also for sampling the transitions, and also for calculating the advantages and returns.
This will be in charge of step 4 and 5 
"""
import torch
import numpy as np

class Buffer:
    def __init__(self, gamma = 0.99):
        # gamma is discount factor
        self.states = []
        self.actions = []
        self.rewards = []
        self.logProbs = []
        self.values = [] # values from critic network
        self.gamma = gamma

    def store(self, state, action, reward, logProb, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.logProbs.append(logProbs)
        self.values.append(value)
        

    # def sample(self):
    #     pass

    def calculate_rtg(self):
        # rtg = r_t + gamma * r_t+1
        # rewardsToGo = a tensor
        return rewardsToGo

    def calculate_advantages(self):
         # A = rewardsToGo - values
        values = torch.tensor(self.values, dtype=torch.float32)
        advantages = rewardsToGo - values
        return advantages
