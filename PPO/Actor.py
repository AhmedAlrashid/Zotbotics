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

class Actor:
    def __init__(self, state_dim = 372, action_dim = 12, lr = 1e-4):
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

    def forward(self, x):
        return self.network(x)