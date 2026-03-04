"""
This is basically the loop class, where we will run the episodes, 
and also where we will collect the transitions, and also where we will update the actor and critic networks.
This will be in charge of step 3, it will basically call a function from buffer to store the transitions.
"""

import torch
import numpy as np
from buffer import Buffer

def episode(environment, actor, critic, buffer, steps=500):
    state = environment.reset()
    total_reward = 0.0

    for _ in range(steps):
        state_tensor = torch.tensor(state, dtype=torch.float32)

        action_tensor, log_prob = actor.get_action(state_tensor)
        value_tensor = critic(state_tensor).squeeze()
        action_numpy = action_tensor.detach().cpu().numpy()
        next_state, reward, done, _ = environment.step(action_numpy)

        buffer.store(
            state=np.array(state, dtype=np.float32),
            action=np.array(action_numpy, dtype=np.float32),
            reward=float(reward),
            logProb=float(log_prob.item()),
            value=float(value_tensor.item())
        )

        total_reward += float(reward)
        state = next_state
        if done:
            state = environment.reset()

    return total_reward