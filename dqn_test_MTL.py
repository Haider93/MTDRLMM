# coding:utf-8

import os
import sys
import gym
import random
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Dense, Average, Add, Maximum
from keras.models import model_from_json, Model
import os.path
from os import path
from keras import backend as K
import json
import time
import numpy as np

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
    def __init__(self, num_actions, num_states_vars, shared_model, task_model):
        self.num_actions = num_actions
        self.epsilon = INITIAL_EPSILON
        self.epsilon_step = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS
        self.t = 0
        # state space information
        self.state_vars = num_states_vars
        ##
        self.shared_model = shared_model
        self.task_model= task_model

        self.model_net = self.build_model()

    def build_model(self):
        inp_layer = self.shared_model.get_layer('shared_input').output
        # out_layer = self.shared_model(inp_layer)
        # aux_hidden = Dense(units=self.state_vars, activation='relu')(out_layer)
        # output_layer_combined = self.task_model(aux_hidden)
        # model = Model(inputs=inp_layer, outputs=output_layer_combined)

        shared_model_out = self.shared_model(inp_layer)
        task_model_out = self.task_model(inp_layer)
        merged_layer = Average()([shared_model_out, task_model_out])
        model = Model(inputs=inp_layer, outputs=merged_layer)
        return model



def load_model(model, weights):
    if(path.exists(model)):
        json_file = open(model, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(weights)
        return loaded_model
    else:
        return False



# load model if exists
#shared_model = load_model("shared_model_spy_dia_xlf_ups_gsk.json", "shared_model_weights_spy_dia_xlf_ups_gsk.h5")
#shared_model = load_model("shared_model_spy_dia_xlf_ups_gsk_amzn_txn.json", "shared_model_weights_spy_dia_xlf_ups_gsk_amzn_txn.h5")
#shared_model = load_model("shared_model_spy_txn_xlf_ups_gsk.json", "shared_model_weights_spy_txn_xlf_ups_gsk.h5")
task_model = load_model("task_model_amzn.json", "task_model_weights_amzn.h5")
agent = Agent(num_actions=NUM_ACTIONS, num_states_vars=NUM_STATE_VARS, shared_model=shared_model, task_model=task_model)

def get_q_values(state=None):
    #c_state = np.ones(NUM_STATE_VARS)
    state = np.asarray(state)
    state = state.reshape(1, NUM_STATE_VARS)
    #q_val = agent.model_net.predict_on_batch(state)
    q_val = agent.task_model.predict_on_batch(state)
    #q_val = agent.shared_model.predict_on_batch(state)
    #print(q_val.flatten().tolist())
    return q_val.flatten().tolist()

#get_q_values()