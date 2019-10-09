import sys
import argparse
import numpy as np
# import tensorflow as tf
# import keras
import gym

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# from keras.models import Sequential
# from keras.layers import Dense, Activation,BatchNormalization
# from keras import backend as K

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from tensorboardX import SummaryWriter

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

class Agent(torch.nn.Module):
    def __init__(self, env):
        super(Agent, self).__init__()

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n
        self.linear1 = nn.Linear(num_states, 16)
        self.linear2 = nn.Linear(16, 16)
        self.linear3 = nn.Linear(16, 16)
        self.linear4 = nn.Linear(16, num_actions)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.softmax(self.linear4(x), dim=1)
        return x

def normalize_returns(Gt):
    Gt -= np.mean(Gt)
    Gt /= np.std(Gt)
    return Gt

def select_action(env, probs, type="prob_sample"):
    if(type=="prob_sample"):
        m = Categorical(probs)
        action = m.sample()
        return action, m.log_prob(action)
    elif(type=="best"):
        return torch.argmax(probs,axis=1)
    elif(type=="random"):
        return env.action_space.sample()

def generate_episode(env, policy, render=False):
    # Generates an episode by executing the current policy in the given env.
    # Returns:
    # - a list of states, indexed by time step
    # - a list of actions, indexed by time step
    # - a list of rewards, indexed by time step
    # TODO: Implement this method.
    states = []
    actions = []
    rewards = []
    log_probs = []

    curr_state = env.reset()
    # curr_state = transform_state(curr_state, env.observation_space.shape[0])
    done = False
    #For our case the len(states) = len(actions)= len(rewards) //VERIFY 
    while(not done):
        if(len(curr_state.shape) == 1):
            curr_state = np.expand_dims(curr_state,0)
        if(torch.cuda.is_available()):
            prob = policy(torch.from_numpy(curr_state)).float().to(device)
        else:
            prob = policy(torch.from_numpy(curr_state)).float()
        action, log_prob = select_action(env, prob)         #VERIFY IF BEST ACTION NEEDS TO BE TAKEN OR RANDOM
        new_state, reward, done, info = env.step(action.item())
        states.append(curr_state)
        actions.append(action)
        rewards.append(reward)
        log_probs.append(log_prob)
        curr_state = new_state
        # curr_state = transform_state(new_state, env.observation_space.shape[0])
    return states, actions, rewards, log_probs

def train(env, policy, optimizer, gamma=1.0):
    # Trains the model on a single episode using REINFORCE.
    # TODO: Implement this method. It may be helpful to call the class
    #       method generate_episode() to generate training data.
    states, actions, rewards, log_probs = generate_episode(env, policy)
    Gt = np.array([rewards[-1]])
    for i in range(len(rewards)-1):
        Gt = np.vstack((rewards[-2-i] + gamma*Gt[0],Gt))
    if(torch.cuda.is_available()):
        Gt = torch.tensor(normalize_returns(Gt)/Gt.shape[0]).squeeze().float().to(device) #Dividing by length to perfrom 1/T step in loss calculation
    else: 
        Gt = torch.tensor(normalize_returns(Gt)/Gt.shape[0]).squeeze().float() #Dividing by length to perfrom 1/T step in loss calculation
    #WRITE POLICY UPDATE STEP
    policy_loss = []
    for i in range(Gt.shape[0]):
        policy_loss.append(Gt[i]*-log_probs[i]) # Since log will be in a value between -inf and 0
    if(torch.cuda.is_available()):
        policy_loss = torch.stack(policy_loss).sum().float().to(device)
    else:
        policy_loss = torch.stack(policy_loss).sum().float()
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
    return policy_loss.item()

def transform_state(state, size):
    return np.reshape(state, [1, size])

def test_agent(env,policy, num_test_epsiodes):
    total_reward = 0
    for e in range(num_test_epsiodes):
        _, _, rewards, _ = generate_episode(env, policy)
        total_reward += sum(rewards)
    print("Testing - Average Reward over %d episodes is %f" %(num_test_epsiodes, total_reward/num_test_epsiodes))
    return int(total_reward/num_test_epsiodes)


def train_agent(policy, env, gamma, num_episodes, optimizer, writer, test_frequency=200, num_test_epsiodes=100, ):
    scores = []
    episodes = []
    for e in range(num_episodes):
        loss = train(env, policy, optimizer, gamma)
        writer.add_scalar("train/Policy Loss", loss, e)
        print("Completed episode %d, with loss: %f"%(e, loss))
        if(e%test_frequency==0):
            score = test_agent(env, policy, num_test_epsiodes)
            writer.add_scalar('test/Reward', score , e)
            scores.append(score)
            episodes.append(e)
    #PLOT ERROR BAR GRAPH NOW

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

def main(args):
    # Parse command-line arguments.
    args = parse_arguments()
    num_episodes = args.num_episodes
    lr = args.lr
    gamma = args.gamma
    render = args.render
    test_frequency = args.test_frequency
    num_test_epsiodes = args.num_test_epsiodes

    writer = SummaryWriter()

    # Create the environment.
    env = gym.make('LunarLander-v2')

    # TODO: Create the model.
    print("Gamma is: ", gamma)
    print("Test Frequency: ", test_frequency)
    print("num_test_epsiodes: ", num_test_epsiodes)

    policy = Agent(env)
    policy.to(device)

    optimizer = optim.Adam(policy.parameters(), lr=lr)

    train_agent(policy=policy, env=env, gamma=gamma, num_episodes=num_episodes, optimizer=optimizer, writer=writer, test_frequency=test_frequency, num_test_epsiodes=num_test_epsiodes)
    # TODO: Train the model using REINFORCE and plot the learning curve.

if __name__ == '__main__':
    main(sys.argv)