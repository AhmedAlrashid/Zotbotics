"""
This class is basically the critic class, which will be responsible for setting up the critic network, 
and the optimizer for the critic network, and also the function to update the critic network.
responsible for step 2 only
"""

import torch
import torch.nn as nn
import torch.optim as optim

num_mapped_out = 256
learning_rate = 1e-4

class Critic(nn.Module):
    def __init__(self, in_dim = 372, lr = learning_rate):
        super(Critic, self).__init__()

# Outputs 1 value V
        self.network = nn.Sequential(
            nn.Linear(in_dim, num_mapped_out),
            nn.ReLU(),
            nn.Linear(num_mapped_out, num_mapped_out),
            nn.ReLU(),
            nn.Linear(num_mapped_out, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        # Return scalar V
        return self.network(state)

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()