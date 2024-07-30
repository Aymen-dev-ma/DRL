import gym
import tensorflow as tf
from model_decon import Model_Decon
from data_handler import DataHandler
import configs

def main():
    opts = configs.model_config

    # Set up the environment
    env = gym.make('CartPole-v1')
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    # Update configuration with environment details
    opts['x_dim'] = observation_space
    opts['a_dim'] = action_space

    # Set up the GPU configuration
    gpu_config = tf.compat.v1.ConfigProto(device_count={'GPU': 1}, allow_soft_placement=False, log_device_placement=False)
    gpu_config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=gpu_config)

    print('Starting processing data ...')

    data = DataHandler(opts)

    print('Starting initializing model ...')
    opts['r_range_upper'] = data.train_r_max
    opts['r_range_lower'] = data.train_r_min
    model = Model_Decon(sess, opts)

    print('Starting training model ...')
    model.train_model(data)

    print('Starting environment interaction ...')
    for episode in range(100):  # Number of episodes
        state = env.reset()
        done = False
        while not done:
            env.render()
            action = model.sample_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            model.update(state, action, reward, next_state, done)
            state = next_state

if __name__ == "__main__":
    main()
