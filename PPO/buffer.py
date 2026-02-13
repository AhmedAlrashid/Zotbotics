"""
This class is basically the buffer class, which will be responsible for storing the transitions, 
and also for sampling the transitions, and also for calculating the advantages and returns.
This will be in charge of step 4 and 5 
"""
import torch
import torch.nn as nn
import numpy as np

class Buffer:
    def __init__(self, gamma = 0.99, batch_size = 12):
        # gamma is discount factor
        self.states = []
        self.actions = []
        self.rewards = []
        self.logProbs = []
        self.values = [] # values from critic network
        self.rtg = None
        self.gamma = gamma
        self.current_state = None
        self.actor = Actor()

    def store(self, state, action, reward, logProb, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.logProbs.append(logProb)
        self.values.append(value)
        

    def sample(self):
        


    def calculate_rtg(self):
        #rewards = torch.tensor(self.rewards, dtype=torch.float32)
        rtg = []
        rewards_calculation = 0
        for i in range(len(self.rewards)):
            rewards_calculation += self.rewards[i] * self.gamma**i
            rtg.append(rewards_calculation)

        rtg = torch.tensor(rtg, dtype=torch.float32)
        self.rtg = rtg
        # rewardsToGo = a tensor
        return rtg

    def calculate_advantages(self):
         # A = rewardsToGo - values
        values = torch.tensor(self.values, dtype=torch.float32)
        advantages = self.rtg - values
        return advantages
