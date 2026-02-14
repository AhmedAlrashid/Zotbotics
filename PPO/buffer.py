"""
This class is basically the buffer class, which will be responsible for storing the transitions, 
and also for sampling the transitions, and also for calculating the advantages and returns.
This will be in charge of step 4 and 5 
"""
import torch
import torch.nn as nn
import numpy as np
from Actor import Actor
from critic import Critic

class Buffer:
    def __init__(self, gamma = 0.99, batch_size = 4):
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
        self.critic = Critic()
        self.batch_size = batch_size # We can change the value of batch size if it makes it more efficient

    def store(self, state, action, reward, logProb, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.logProbs.append(logProb)
        self.values.append(value)
        

    def sample(self):
        dist = self.actor(self.current_state)
        value = self.critic(self.current_state)
        all_actions = dist.sample(self.batch_size)
        
        initial_reward = 0
        for action in all_actions:
            self.store(self.current_state, action, initial_reward, dist[action], value)




        


    def calculate_rtg(self):
        rtg = 0
        rtgArray = []
        for reward in reversed(self.rewards):
            rtg = reward + self.gamma * rtg
            rtgArray.append(rtg)
        rtgArray.reverse()
        rtgTensor = torch.tensor(rtgArray, dtype=torch.float32)
        return rtgTensor

    def calculate_advantages(self):
         # A = rewardsToGo - values
        values = torch.tensor(self.values, dtype=torch.float32)
        rtg = self.calculate_rtg()
        advantages = rtg - values
        return advantages
