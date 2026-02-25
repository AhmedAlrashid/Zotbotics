"""
This is basically the loop class, where we will run the episodes, 
and also where we will collect the transitions, and also where we will update the actor and critic networks.
This will be in charge of step 3, it will basically call a function from buffer to store the transitions.
"""

import torch
import numpy as np
from buffer import Buffer

def episode(environment, actor, critic, steps=500, gamma=0.99):
    buffer = Buffer(gamma=gamma)

    state = environment.reset()
    total_reward = 0.0

    for _ in range(steps):
        state_tensor = torch.tensor(state, dtype=torch.float32)

        action_tensor = actor(state_tensor)
        value_tensor = critic(state_tensor).squeeze()
        action_numpy = action_tensor.detach().cpu().numpy()
        next_state, reward, done, _ = environment.step(action_numpy)

        buffer.store(
            state=np.array(state, dtype=np.float32),
            action=np.array(action_numpy, dtype=np.float32),
            reward=float(reward),
            logProb=0.0,
            value=float(value_tensor.item())
        )

        total_reward += float(reward)
        state = next_state
        if done:
            state = environment.reset()

    returns_to_go = buffer.calculate_rtg()
    advantages = buffer.calculate_advantages()

    # Convert into tensor
    states_tensor = torch.tensor(np.array(buffer.states), dtype=torch.float32)
    actions_tensor = torch.tensor(np.array(buffer.actions), dtype=torch.float32)

    # Update Critic
    predicted_values = critic(states_tensor).squeeze()  # (T,)
    critic_loss = torch.mean((returns_to_go - predicted_values) ** 2)
    critic.update(critic_loss)

    # Update Actor
    predicted_actions = actor(states_tensor)  # (T, 12)

    actor_loss = -torch.mean(
        advantages.detach() * torch.sum(predicted_actions * actions_tensor, dim=1)
    )
    actor.update(actor_loss)

    return total_reward