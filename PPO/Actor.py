"""
This class will basically be the actor class, which will
be responsible for setting up the actor network, 
and the optimizer for the actor network, and also the function to update the actor network.
responsible for step 1 only 
"""

"""
This should have 372 input nodes,
and 12 ouput nodes"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

"""I believe 256 is efficient here but if you guys like feel free to amplify by another power of two
or so if the agent is failing to learn even for the simplest tasks."""
num_mapped_out = 256
learning_rate = 1e-4

class Actor(nn.Module):
    def __init__(self, in_dim = 372, out_dim = 12, lr = learning_rate): # 372 inputs, 12 outputs (Lidar, servo motors)
        super(Actor, self).__init__() # Inherit init from parent. Allows for PyTorch features to run.
        
        # Initialize the Actor network.
        self.network = nn.Sequential(
            nn.Linear(in_dim, num_mapped_out), # Map into the value of num_mapped_out.
            nn.ReLU(), # Activation func.
            nn.Linear(num_mapped_out, num_mapped_out),
            nn.ReLU(),
            nn.Linear(num_mapped_out, out_dim), # Maps into the 12 outputs.
            nn.Tanh() # --> You can also use nn.Softmax() here, but I think nn.Tanh() is better for continuous actions (efficient for servos).
        )
        
        # Add log_std parameter for action distribution
        self.log_std = nn.Parameter(torch.zeros(out_dim))
        
        self.optimizer = optim.Adam(self.parameters(), lr = lr) # Update the weights with the learning rate.

    def forward(self, state): # Passes a state to get an action distribution.
        mean = self.network(state)
        std = torch.exp(self.log_std)
        return Normal(mean, std)
    
    def get_action(self, state):
        dist = self.forward(state)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob
    
    def update(self, loss): # Updates the Actor network.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()