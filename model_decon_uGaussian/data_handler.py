import numpy as np
import gym

class DataHandler:
    def __init__(self, opts):
        self.load_data(opts)

    def load_data(self, opts):
        if opts['dataset'] == 'gym':
            self.load_gym_data(opts)
        else:
            raise ValueError('Dataset not supported!')

    def load_gym_data(self, opts):
        env = gym.make('CartPole-v1')
        num_episodes = 100
        max_steps = 200

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for episode_index in range(num_episodes):
            state = env.reset()
            episode_states = []
            episode_actions = []
            episode_rewards = []
            episode_next_states = []
            episode_dones = []

            for step_index in range(max_steps):
                action = env.action_space.sample()
                next_state, reward, done, truncated, info = env.step(action)
                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards.append(reward)
                episode_next_states.append(next_state)
                episode_dones.append(done)
                state = next_state
                if done or truncated:
                    break

            # Padding for uniform length
            while len(episode_states) < max_steps:
                episode_states.append(np.zeros_like(state))
                episode_actions.append(0)
                episode_rewards.append(0.0)
                episode_next_states.append(np.zeros_like(state))
                episode_dones.append(True)

            states.append(episode_states)
            actions.append(episode_actions)
            rewards.append(episode_rewards)
            next_states.append(episode_next_states)
            dones.append(episode_dones)

        self.states = np.array(states, dtype=np.float32)
        self.actions = np.array(actions, dtype=np.int32)
        self.rewards = np.array(rewards, dtype=np.float32)
        self.next_states = np.array(next_states, dtype=np.float32)
        self.dones = np.array(dones, dtype=bool)

        self.train_r_max = np.max(self.rewards)
        self.train_r_min = np.min(self.rewards)
