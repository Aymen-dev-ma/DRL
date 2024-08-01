import gym
import torch
import numpy as np

class GymDataHandler:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.state = self.env.reset()

    def get_initial_state(self):
        return torch.tensor(self.state, dtype=torch.float32).flatten()

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        if done:
            state = self.env.reset()
        return state.flatten(), reward, done, info

    def sample_action(self):
        return self.env.action_space.sample()

    def generate_batch(self, batch_size):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for _ in range(batch_size):
            action = self.sample_action()
            next_state, reward, done, _ = self.step(action)
            states.append(self.state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            self.state = next_state
        try:
            states = torch.tensor(np.array(states), dtype=torch.float32)
            actions = torch.tensor(np.array(actions), dtype=torch.float32).unsqueeze(1)
            rewards = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
            dones = torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1)
        except Exception as e:
            print(f"Error in generate_batch: {e}")
            print(f"states: {states}")
            print(f"actions: {actions}")
            print(f"rewards: {rewards}")
            print(f"next_states: {next_states}")
            print(f"dones: {dones}")
            raise e
        return states, actions, rewards, next_states, dones
