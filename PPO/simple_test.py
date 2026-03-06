"""
Simple test script for your PPO implementation
"""

import torch
import numpy as np
from PPO_agent import PPOAgent

class SimpleTestEnvironment:
    """Simple test environment - reach target position"""
    def __init__(self):
        self.state_dim = 372  # Match your robot's state size
        self.action_dim = 12  # Match your robot's action size
        self.max_steps = 100
        self.current_step = 0
        self.reset()
    
    def reset(self):
        self.current_step = 0
        # Random starting position (simplified state)
        self.position = np.random.uniform(-5, 5, 2)  # [x, y]
        self.target = np.random.uniform(-3, 3, 2)    # [target_x, target_y]
        
        # Create full 372-dimensional state (padding with zeros)
        state = np.zeros(self.state_dim, dtype=np.float32)
        state[0:2] = self.position  # Position
        state[2:4] = self.target    # Target
        # Fill rest with small random noise (simulating sensor readings)
        state[4:] = np.random.normal(0, 0.1, self.state_dim - 4)
        
        return state
    
    def step(self, action):
        self.current_step += 1
        
        # Use first 2 actions as movement commands
        movement = np.clip(action[:2], -1, 1) * 0.5  # Limit movement speed
        self.position += movement
        
        # Calculate reward (negative distance to target)
        distance = np.linalg.norm(self.position - self.target)
        reward = -distance  # Closer = higher reward
        
        # Bonus for getting very close
        if distance < 0.5:
            reward += 10
        
        # Small penalty for large actions (energy efficiency)
        action_penalty = -0.01 * np.sum(np.square(action))
        reward += action_penalty
        
        # Done conditions
        done = (self.current_step >= self.max_steps) or (distance < 0.1)
        
        # Create next state
        next_state = np.zeros(self.state_dim, dtype=np.float32)
        next_state[0:2] = self.position
        next_state[2:4] = self.target
        next_state[4:] = np.random.normal(0, 0.1, self.state_dim - 4)
        
        return next_state, reward, done, {}

def test_components():
    """Test individual components"""
    print("🔧 Testing components...")
    
    # Test environment
    env = SimpleTestEnvironment()
    state = env.reset()
    print(f"✅ Environment: State shape {state.shape}")
    
    # Test a step
    action = np.random.randn(12)
    next_state, reward, done, _ = env.step(action)
    print(f"✅ Environment step: Reward {reward:.3f}, Done {done}")
    
    # Test networks
    from Actor import Actor
    from critic import Critic
    
    actor = Actor()
    critic = Critic()
    
    state_tensor = torch.tensor(state, dtype=torch.float32)
    action, log_prob = actor.get_action(state_tensor)
    value = critic(state_tensor)
    
    print(f"✅ Actor: Action shape {action.shape}, LogProb {log_prob.item():.3f}")
    print(f"✅ Critic: Value {value.item():.3f}")
    
    return True

def quick_test(episodes=20):
    """Quick PPO training test"""
    print(f"🚀 Quick PPO test ({episodes} episodes)...")
    
    env = SimpleTestEnvironment()
    
    # Create agent with small settings for quick test
    agent = PPOAgent(
        environment=env,
        episodes_per_update=5,  # Small for quick test
        epochs=2,               # Fewer epochs for speed
        batch_size=32           # Smaller batches
    )
    
    print("Starting training...")
    rewards = []
    
    try:
        # Short training run
        for update in range(4):  # Just 4 updates
            avg_reward = agent.collect_experience()
            agent.update_policy()
            agent.buffer.clear()
            
            rewards.append(avg_reward)
            print(f"Update {update}: Avg Reward = {avg_reward:.2f}")
        
        print(f"✅ Training completed! Final reward: {rewards[-1]:.2f}")
        
        # Check improvement
        if len(rewards) >= 2:
            improvement = rewards[-1] - rewards[0]
            print(f"📈 Improvement: {improvement:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def longer_test(updates=50):
    """Longer training to see learning"""
    print(f"🎯 Longer PPO test ({updates} updates)...")
    
    env = SimpleTestEnvironment()
    agent = PPOAgent(environment=env, episodes_per_update=10)
    
    print("Training (this may take a minute)...")
    
    try:
        trained_actor, trained_critic = agent.train(num_updates=updates)
        print("✅ Longer training completed successfully!")
        
        # Test the trained agent
        print("🎮 Testing trained agent...")
        test_rewards = []
        
        for test_ep in range(5):
            state = env.reset()
            total_reward = 0
            
            for step in range(100):
                state_tensor = torch.tensor(state, dtype=torch.float32)
                with torch.no_grad():
                    action, _ = trained_actor.get_action(state_tensor)
                    action = action.numpy()
                
                state, reward, done, _ = env.step(action)
                total_reward += reward
                
                if done:
                    break
            
            test_rewards.append(total_reward)
            print(f"   Test episode {test_ep}: {total_reward:.2f}")
        
        avg_test_reward = np.mean(test_rewards)
        print(f"🏆 Average test reward: {avg_test_reward:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Longer training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 PPO Implementation Test")
    print("="*40)
    
    # Test 1: Components
    if not test_components():
        print("❌ Component test failed!")
        exit(1)
    
    print()
    
    # Test 2: Quick training
    if not quick_test():
        print("❌ Quick test failed!")
        exit(1)
    
    print()
    
    # Test 3: Ask for longer test
    choice = input("🎯 Run longer training test? (y/n): ").lower().strip()
    if choice == 'y':
        longer_test()
    
    print("\n🎉 PPO implementation is working!")
    print("\n📝 To use with your robot:")
    print("   1. Replace SimpleTestEnvironment with your robot environment")
    print("   2. Make sure your environment returns 372D states")  
    print("   3. Make sure your environment accepts 12D actions")
    print("   4. Implement proper reward function for your task")