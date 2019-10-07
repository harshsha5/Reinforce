import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
import gym
import ipdb

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation,BatchNormalization
from tensorboardX import SummaryWriter
from keras import backend as K


class Reinforce(object):
    # Implementation of the policy gradient method REINFORCE.

    def __init__(self, model, lr):
        self.model = model
        model.compile(loss="categorical_crossentropy",
             optimizer=tf.train.AdamOptimizer(learning_rate=lr))
        self.writer = SummaryWriter()

        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here.

    def train(self, env, gamma=1.0):
        # Trains the model on a single episode using REINFORCE.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.
        states, actions, rewards = self.generate_episode(env)
        Gt = np.array([rewards[-1]])
        for i in range(len(rewards)-1):
            Gt = np.vstack((rewards[-2-i] + gamma*Gt[0],Gt))
        Gt = normalize_returns(Gt)
        #WRITE POLICY UPDATE STEP

        return

    def generate_episode(self, env, render=False):
        # Generates an episode by executing the current policy in the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        # TODO: Implement this method.
        states = []
        actions = []
        rewards = []

        curr_state = env.reset()
        curr_state = transform_state(curr_state,env.observation_space.shape[0])
        done = False
        #For our case the len(states) = len(actions)= len(rewards) //VERIFY 
        while(not done):
            action_probab = self.model.predict_on_batch(curr_state)
            action = select_action(env,action_probab,"best")            #VERIFY IF BEST ACTION NEEDS TO BE TAKEN OR RANDOM
            new_state, reward, done, info = env.step(action)
            states.append(curr_state)
            rewards.append(reward)
            actions.append(action)
            curr_state = transform_state(new_state,env.observation_space.shape[0])
        return states, actions, rewards

def normalize_returns(Gt):
    Gt -= np.mean(Gt)
    Gt /= np.std(Gt)
    return Gt

def select_action(env,action_probab,type="best"):
    if(type=="best"):
        return np.argmax(action_probab,axis=1)[0]
    elif(type=="random"):
        return env.action_space.sample()

def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float, default=5e-4, help="The learning rate.")
    parser.add_argument('--gamma', dest='gamma', type=float, default=1, help="The discount factor")
    parser.add_argument('--test_frequency', dest='test_frequency', type=int, default=200, help="After how many policies do I test the model")
    parser.add_argument('--num_test_epsiodes', dest='num_test_epsiodes', type=int, default=100, help="For how many policies do I test the model")
    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',action='store_true',help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',action='store_false',help="Whether to render the environment.")
    parser.set_defaults(render=False)

    return parser.parse_args()

def make_model(env):
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n
    print("Number of states: ",num_states)
    print("Number of actions: ",num_actions)
    ### SEE IF BIAS NEEDS TO BE MADE FALSE
    model = Sequential()
    model.add(Dense(16, activation='relu', input_dim=num_states,bias_initializer='zeros',kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform', seed=None)))
    model.add(BatchNormalization())
    model.add(Dense(16, activation='relu',bias_initializer='zeros',kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform', seed=None)))
    model.add(BatchNormalization())
    model.add(Dense(16, activation='relu',bias_initializer='zeros',kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform', seed=None)))
    model.add(BatchNormalization())
    model.add(Dense(num_actions, activation='softmax', bias_initializer='zeros',kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform', seed=None)))
    # print(model.summary())
    # config = model.get_config()
    # print(config)
    return model

def transform_state(state,size):
    return np.reshape(state, [1, size])

def test_agent(env,agent,num_test_epsiodes):
    total_reward = 0
    for e in range(num_test_epsiodes):
        _, _, rewards = agent.generate_episode(env)
        total_reward += sum(rewards)
    print("Reward is: ",int(total_reward/num_test_epsiodes))
    return int(total_reward/num_test_epsiodes)


def train_agent(agent,env,gamma,num_episodes,test_frequency = 200, num_test_epsiodes = 100):
    scores = []
    episodes = []
    for e in range(num_episodes):
        agent.train(env,gamma)
        if(e%test_frequency==0):
            score = test_agent(env,agent,num_test_epsiodes)
            agent.writer.add_scalar('Reward VS Episode', score , e)
            scores.append(score)
            episodes.append(e)


def main(args):
    # Parse command-line arguments.
    args = parse_arguments()
    num_episodes = args.num_episodes
    lr = args.lr
    gamma = args.gamma
    render = args.render
    test_frequency = args.test_frequency
    num_test_epsiodes = args.num_test_epsiodes

    # Create the environment.
    env = gym.make('LunarLander-v2')

    # TODO: Create the model.
    print("Gamma is: ",gamma)
    print("Test Frequency: ",test_frequency)
    print("num_test_epsiodes: ",num_test_epsiodes)
    model = make_model(env)
    agent = Reinforce(model,lr)
    train_agent(agent,env,gamma,num_episodes,test_frequency,num_test_epsiodes)
    # TODO: Train the model using REINFORCE and plot the learning curve.



if __name__ == '__main__':
    main(sys.argv)
