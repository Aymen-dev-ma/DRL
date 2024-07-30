#########################
## Author: Chaochao Lu ##
#########################
import numpy as np
import logging

class DataHandler(object):
    def __init__(self, opts):
        self.train_num = None
        self.validation_num = None
        self.test_num = None
        self.x_train = None
        self.a_train = None
        self.r_train = None
        self.mask_train = None
        self.x_validation = None
        self.a_validation = None
        self.r_validation = None
        self.mask_validation = None
        self.x_test = None
        self.a_test = None
        self.r_test = None
        self.mask_test = None
        self.train_r_max = None
        self.train_r_min = None
        self.load_data(opts)

    def load_data(self, opts):
        if opts['dataset'] == 'dataset_name':
            self.load_dataset(opts)
        else:
            logging.error(opts['dataset'] + ' cannot be found.')

    def load_dataset(self, opts):
        # Load the dataset from the npz file
        data = np.load('gym_data.npz')
        self.x_train = data['observations']
        self.a_train = data['actions']
        self.r_train = data['rewards']
        self.mask_train = data['dones']

        # Split data into training, validation, and test sets
        split_1 = int(0.8 * len(self.x_train))
        split_2 = int(0.9 * len(self.x_train))
        self.x_validation = self.x_train[split_1:split_2]
        self.a_validation = self.a_train[split_1:split_2]
        self.r_validation = self.r_train[split_1:split_2]
        self.mask_validation = self.mask_train[split_1:split_2]

        self.x_test = self.x_train[split_2:]
        self.a_test = self.a_train[split_2:]
        self.r_test = self.r_train[split_2:]
        self.mask_test = self.mask_train[split_2:]

        self.x_train = self.x_train[:split_1]
        self.a_train = self.a_train[:split_1]
        self.r_train = self.r_train[:split_1]
        self.mask_train = self.mask_train[:split_1]

        self.train_num = self.x_train.shape[0]
        self.validation_num = self.x_validation.shape[0]
        self.test_num = self.x_test.shape[0]

        self.train_r_max = np.amax(self.r_train)
        self.train_r_min = np.amin(self.r_train)
