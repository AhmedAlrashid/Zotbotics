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
        # Invalidate cached rtg when new data added
        self.rtg = None
        
    def get_all_data(self):
        """Return all stored data as tensors for PPO update"""
        states = torch.tensor(np.array(self.states), dtype=torch.float32)
        actions = torch.tensor(np.array(self.actions), dtype=torch.float32)
        old_log_probs = torch.tensor(self.logProbs, dtype=torch.float32)
        advantages = self.calculate_advantages()
        returns = self.calculate_rtg()
        return states, actions, old_log_probs, advantages, returns
    
    def get_batches(self):
        """Yield mini-batches for more stable training - OPTIMIZED"""
        total_size = len(self.states)
        indices = torch.randperm(total_size)
        
        # Convert lists to numpy arrays ONCE (much faster)
        states_array = np.array(self.states)
        actions_array = np.array(self.actions)
        logprobs_array = np.array(self.logProbs)
        
        # Pre-calculate advantages and returns
        advantages = self.calculate_advantages()
        returns = self.calculate_rtg()
        
        for start in range(0, total_size, self.batch_size):
            end = min(start + self.batch_size, total_size)
            batch_indices = indices[start:end]
            
            # Use numpy indexing then convert to tensor (MUCH faster)
            batch_states = torch.tensor(states_array[batch_indices], dtype=torch.float32)
            batch_actions = torch.tensor(actions_array[batch_indices], dtype=torch.float32)
            batch_old_log_probs = torch.tensor(logprobs_array[batch_indices], dtype=torch.float32)
            batch_advantages = advantages[batch_indices]
            batch_returns = returns[batch_indices]
            
            yield batch_states, batch_actions, batch_old_log_probs, batch_advantages, batch_returns

    def clear(self):
        """Clear all stored data for next collection phase"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.logProbs.clear()
        self.values.clear()
        self.rtg = None
        self.current_state = None

    def sample(self):
        # Legacy method - PPO doesn't use random sampling
        # Keeping for compatibility
        pass


    def calculate_rtg(self):
        # Use cached version if available for efficiency
        if self.rtg is not None:
            return self.rtg
            
        rtg = 0
        rtgArray = []
        for reward in reversed(self.rewards):
            rtg = reward + self.gamma * rtg
            rtgArray.append(rtg)
        rtgArray.reverse()
        self.rtg = torch.tensor(rtgArray, dtype=torch.float32)
        return self.rtg

    def calculate_advantages(self):
         # A = rewardsToGo - values
        values = torch.tensor(self.values, dtype=torch.float32)
        rtg = self.calculate_rtg()
        advantages = rtg - values
        return advantages
