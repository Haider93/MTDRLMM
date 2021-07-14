# coding:utf-8

import os
import sys
import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Sequential
from keras.layers import Conv1D, Flatten, Dense, LSTM, Dropout, MaxPooling1D, Conv2D, MaxPooling2D, Activation, Input, InputLayer
from keras.models import model_from_json, Model
from keras.optimizers import Adam, SGD, RMSprop
from keras.losses import mean_squared_error
import os.path
from os import path
from keras import backend as K
import json
import time
from dkeras import dkeras

ENV_NAME = 'Breakout-v0'  # Environment name
FRAME_WIDTH = 84  # Resized frame width
FRAME_HEIGHT = 84  # Resized frame height
NUM_EPISODES = 10  # Number of episodes the agent plays
NUM_ACTIONS = 9 # Number of discrete actions
NUM_STATE_VARS = 8 # Number of state variables or dimension of the state space
STATE_LENGTH = 4  # Number of most recent frames to produce the input to the network
GAMMA = 0.975  # Discount factor
EXPLORATION_STEPS = 1000000  # Number of steps over which the initial value of epsilon is linearly annealed to its final value
INITIAL_EPSILON = 1.0  # Initial value of epsilon in epsilon-greedy
FINAL_EPSILON = 0.1  # Final value of epsilon in epsilon-greedy
INITIAL_REPLAY_SIZE = 20000  # Number of steps to populate the replay memory before training starts
NUM_REPLAY_MEMORY = 400000  # Number of replay memory the agent uses for training
BATCH_SIZE = 32  # Mini batch size
TARGET_UPDATE_INTERVAL = 10000  # The frequency with which the target network is updated
TRAIN_INTERVAL = 4  # The agent selects 4 actions between successive updates
LEARNING_RATE = 0.00025  # Learning rate used by RMSProp
MOMENTUM = 0.95  # Momentum used by RMSProp
MIN_GRAD = 0.01  # Constant added to the squared gradient in the denominator of the RMSProp update
SAVE_INTERVAL = 300000  # The frequency with which the network is saved
NO_OP_STEPS = 30  # Maximum number of "do nothing" actions to be performed by the agent at the start of an episode
LOAD_NETWORK = False
TRAIN = True
SAVE_NETWORK_PATH = 'saved_networks/' + ENV_NAME
SAVE_SUMMARY_PATH = 'summary/' + ENV_NAME
NUM_EPISODES_AT_TEST = 30  # Number of episodes the agent plays at test time


class Agent():
    def __init__(self, num_actions, num_states_vars, gamma):
        self.num_actions = num_actions
        self.epsilon = INITIAL_EPSILON
        self.epsilon_step = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS
        self.t = 0
        self.gamma = gamma
        # state space information
        self.state_vars = num_states_vars
        # self.current_state = current_state
        # self.next_state = next_state
        # self.reward = reward
        # self.last_action = last_action
        self.q_values = np.zeros(self.num_actions)

        # Parameters used for summary
        self.total_reward = 0
        self.total_q_max = 0
        self.total_loss = 0
        self.duration = 0
        self.episode = 0

        # Create replay memory
        self.replay_memory = deque(maxlen=2000)

        ##main q network
        self.model_net = self.build_network()

        ##auxiliary q network
        #self.target_model = self.build_network()

        #self.update_target_network()

        print("constructor called.")


    def update_target_network(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model_net.get_weights())

    def remember(self, state, action, reward, next_state, done=False):
        self.replay_memory.append((state, action, reward, next_state, done))

    def build_network(self):
        ##shared model
        # shared_input_layer = Input(shape=(self.state_vars,), name='shared_input')
        # shared_hidden_1 = Dense(64, activation='relu', name='shared_hidden_1')(shared_input_layer)
        # shared_hidden_2 = Dense(64, activation='relu', name='shared_hidden_1')(shared_hidden_1)
        # shared_output = Dense(self.num_actions, activation='linear', name='shared_output')(shared_hidden_2)
        # model = Model(inputs=shared_input_layer, outputs=shared_output, name='shared_model')
        # model.compile(loss="mean_squared_error",
        #               optimizer=Adam(lr=0.001))

        #
        model = Sequential()
        model.add(InputLayer(input_shape=(self.state_vars,), name='shared_input'))
        model.add(Dense(64))
        model.add(Activation('relu'))
        
        model.add(Dense(64))
        model.add(Activation('relu'))
        
        model.add(Dense(self.num_actions, activation='linear', name='shared_output'))
        model.compile(loss="mean_squared_error",
                      optimizer=Adam(lr=0.001))

        return model

    def save_model(self, model, weights):
        # save model to json
        model_json = self.model_net.to_json()
        with open(model, "w") as json_file:
            json_file.write(model_json)  #
        self.model_net.save_weights(weights)

    def save_weights(self, weights):
        self.model_net.save_weights(weights)

    def update_model_network(self, batch_size):
        # self.q_values[last_action] = reward + GAMMA * np.max(self.model_net.predict_on_batch(next_state))
        # target = self.q_values
        # target = np.array([target])
        # #self.model_net.fit(current_state, target, epochs=1)
        # self.model_net.train_on_batch(current_state, target)

        if(batch_size > len(self.replay_memory)):
            batch_size = len(self.replay_memory)
        minibatch = random.sample(self.replay_memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model_net.predict_on_batch(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = self.model_net.predict_on_batch(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(Q_future)
                #self.q_values = target[0].flatten()
            self.model_net.train_on_batch(state, target)
            #print("weights::", self.model_net.get_layer('shared_hidden_1').get_weights())
        return



agent = Agent(num_actions=NUM_ACTIONS, num_states_vars=NUM_STATE_VARS, gamma=GAMMA)
agent.save_model("shared_model_spy_dia_xlf_ups_gsk_amzn_txn.json", "shared_model_weights_spy_dia_xlf_ups_gsk_amzn_txn.h5")

def train_model(curr_state=None, next_state=None, reward=None,action=None):
    # arguments from c++ code
    c_state = np.asarray(curr_state)
    #print("Current state::", c_state)
    c_state = c_state.reshape(1, NUM_STATE_VARS)
    n_state = np.asarray(next_state)
    #print("In train model, next state::", n_state)
    n_state = n_state.reshape(1, NUM_STATE_VARS)
    rew = reward
    last_action = int(action)

    ##test params
    # c_state = np.ones(NUM_STATE_VARS)
    # c_state = c_state.reshape(1, NUM_STATE_VARS)
    # n_state = np.ones(NUM_STATE_VARS)
    # n_state = n_state.reshape(1, NUM_STATE_VARS)
    # rew = 1.5
    # last_action = int(5.0)

    agent.remember(c_state, last_action, rew, n_state)
    agent.update_model_network(batch_size=5)
    agent.save_weights("shared_model_weights_spy_dia_xlf_ups_gsk_amzn_txn.h5")
    #self.update_target_network()
    # print("Train model function called.")
    return agent.q_values.flatten().tolist()

def get_q_values(state=None):
    state = np.asarray(state)
    state = state.reshape(1, NUM_STATE_VARS)
    q_val = agent.model_net.predict_on_batch(state)
    #print("call model function.")
    return q_val.flatten().tolist()

#train_model()
