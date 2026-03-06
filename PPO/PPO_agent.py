"""Essentialy, this is like the main.py or main file
for the PPO Agent, we will basically call all of our functions and classes from here,
We will puzle the pieces together here, and this is where we will train our agent.
This will be in charge of steps 6 and onwards. """

import torch
import numpy as np
from Actor import Actor
from critic import Critic
from buffer import Buffer
from episode import episode

class PPOAgent:
    def __init__(self, n_inputs,n_actions, gamma=0.99, clip_ratio=0.2, epochs=4, batch_size=64, episodes_per_update=10):
        self.actor = Actor(in_dim=n_inputs, out_dim=n_actions)
        self.critic = Critic(in_dim=n_inputs)
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        self.batch_size = batch_size
        self.episodes_per_update = episodes_per_update
        # Use your enhanced buffer with all features
        self.buffer = Buffer(gamma=gamma, batch_size=batch_size)
        self.n_inputs = n_inputs
        self.n_actions = n_actions
    
    def choose_action(self, obs):
        """Choose action using the actor network"""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        
        # Get action distribution from actor
        dist = self.actor(obs_tensor)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Get value from critic
        value = self.critic(obs_tensor).squeeze()
        
        return action.detach().numpy().squeeze(), log_prob.detach(), value.detach()
    
    def remember(self, obs, action, log_prob, value, reward, done):
        """Store experience in buffer"""
        self.buffer.store(obs, action, reward, log_prob, value)
    
    def learn(self):
        """Learn from collected experiences"""
        if len(self.buffer.states) > 0:
            self.update_policy()
            self.buffer.clear()
    
    def save_models(self, actor_path="actor_model.pth", critic_path="critic_model.pth"):
        """Save actor and critic models"""
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        print(f"Models saved to {actor_path} and {critic_path}")
    
    def load_models(self, actor_path="actor_model.pth", critic_path="critic_model.pth"):
        """Load actor and critic models"""
        try:
            self.actor.load_state_dict(torch.load(actor_path))
            self.critic.load_state_dict(torch.load(critic_path))
            print(f"Models loaded from {actor_path} and {critic_path}")
        except FileNotFoundError as e:
            print(f"Model files not found: {e}")

    def collect_experience(self):
        """Collect experience using your modular episode function"""
        total_rewards = []
        
        for _ in range(self.episodes_per_update):
            episode_reward = episode(self.environment, self.actor, self.critic, self.buffer)
            total_rewards.append(episode_reward)
        
        return np.mean(total_rewards)

    def update_policy(self):
        """PPO policy update with clipping using your buffer's features"""
        # Get all collected data
        states, actions, old_log_probs, advantages, returns = self.buffer.get_all_data()
        
        # Normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Multiple epochs of optimization (PPO's key feature)
        for epoch in range(self.epochs):
            #Use mini-batches (your batch_size feature)
            for batch_states, batch_actions, batch_old_log_probs, batch_advantages, batch_returns in self.buffer.get_batches():
                
                # Update Actor (separate forward pass)
                dist = self.actor(batch_states)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1) if batch_actions.dim() > 1 else dist.log_prob(batch_actions)
                
                # PPO clipped objective
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                
                actor_loss = -torch.min(
                    ratio * batch_advantages,
                    clipped_ratio * batch_advantages
                ).mean()
                
                self.actor.update(actor_loss)
                
                # Update Critic (separate forward pass)
                values = self.critic(batch_states).squeeze()
                critic_loss = torch.mean((batch_returns - values) ** 2)
                
                self.critic.update(critic_loss)


    def train(self, num_updates=1000):
        """Train the PPO agent using your modular design"""
        for update in range(num_updates):
            # Collect experience using your episode function
            avg_episode_reward = self.collect_experience()
            
            # Update policy with PPO clipping
            self.update_policy()
            
            # Log progress
            if update % 10 == 0:
                print(f"Update {update}, Avg Episode Reward: {avg_episode_reward:.2f}")
            
            # Clear buffer for next collection (your buffer's feature)
            self.buffer.clear()
        
        return self.actor, self.critic