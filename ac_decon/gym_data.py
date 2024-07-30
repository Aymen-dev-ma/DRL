import gym
import numpy as np

# Create the Gym environment
env = gym.make('CartPole-v1')

# Define the number of episodes and the maximum steps per episode
num_episodes = 1000
max_steps = 200

# Initialize lists to hold the data
observations = []
actions = []
rewards = []
dones = []

# Collect data
for episode in range(num_episodes):
    obs = env.reset()
    for step in range(max_steps):
        action = env.action_space.sample()  # Take a random action
        next_obs, reward, done, truncated, info = env.step(action)  # Updated line
        
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        dones.append(done or truncated)  # Combine done and truncated into one

        obs = next_obs
        
        if done or truncated:
            break

# Convert lists to numpy arrays
observations = np.array(observations)
actions = np.array(actions)
rewards = np.array(rewards)
dones = np.array(dones)

# Save the data as numpy arrays
np.savez('gym_data.npz', observations=observations, actions=actions, rewards=rewards, dones=dones)
