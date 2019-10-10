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
import time

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

class Agent(torch.nn.Module):
    def __init__(self, env, hidden_units, output=None):
        super(Agent, self).__init__()

        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.n if output==None else output
        self.linear1 = nn.Linear(self.num_states, hidden_units)
        self.linear2 = nn.Linear(hidden_units, hidden_units)
        self.linear3 = nn.Linear(hidden_units, hidden_units)
        self.linear4 = nn.Linear(hidden_units, self.num_actions)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.softmax(self.linear4(x), dim=1)
        return x
    
    def init_weights(self, m):
        if type(m) == nn.Linear:
            alpha = np.sqrt(3/((self.num_states+self.num_actions)/2))
            m.bias.data.fill_(0)
            m.weight.data.uniform_(-alpha,alpha)

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
        prob = policy(torch.from_numpy(curr_state).float().to(device))
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
    Gt = torch.tensor(normalize_returns(Gt)/Gt.shape[0]).squeeze().float().to(device) #Dividing by length to perfrom 1/T step in loss calculation
    #WRITE POLICY UPDATE STEP
    policy_loss = []
    for i in range(Gt.shape[0]):
        policy_loss.append(Gt[i]*-log_probs[i]) # Since log will be in a value between -inf and 0
    policy_loss = torch.stack(policy_loss).sum().float().to(device)
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
    return policy_loss.item()

def transform_state(state, size):
    return np.reshape(state, [1, size])

def test_agent(env, policy, num_test_epsiodes):
    rewards=[]
    for e in range(num_test_epsiodes):
        _, _, episode_rewards, _ = generate_episode(env, policy)
        rewards.append(sum(episode_rewards))
    rewards = np.array(rewards)
    print("Testing - %d episodes mean %f & std deviation %f" %(num_test_epsiodes, rewards.mean(), rewards.std()))
    return rewards.mean(), rewards.std()


def train_agent(policy, env, optimizer, writer, args):
    scores = []
    episodes = []
    stds = []
    for e in range(args.num_episodes):
        loss = train(env, policy, optimizer, args.gamma)
        writer.add_scalar("train/Policy Loss", loss, e)
        print("Completed episode %d, with loss: %f"%(e, loss))
        if(e % args.test_frequency==0):
            score, std = test_agent(env, policy, args.num_test_epsiodes)
            writer.add_scalar('test/Reward', score , e)
            scores.append(score)
            episodes.append(e)
            stds.append(std)
    return episodes, scores, stds
    #PLOT ERROR BAR GRAPH NOW

def plot_error_bar(episodes, means, stds):
    fig = plt.figure()
    plt.errorbar(episodes, means, stds)
    plt.savefig("mean_std.png")

def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float, default=5e-4, help="The learning rate.")
    parser.add_argument('--gamma', dest='gamma', type=float, default=1, help="The discount factor")
    parser.add_argument('--test_frequency', dest='test_frequency', type=int, default=200, help="After how many policies do I test the model")
    parser.add_argument('--num_test_epsiodes', dest='num_test_epsiodes', type=int, default=100, help="For how many policies do I test the model")
    parser.add_argument('--hidden_units', dest='hidden_units', type=int, default=16, help="Number of Hidden units in the linear layer")
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

    timestr = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter("reinforce_"+timestr)

    # Create the environment.
    env = gym.make('LunarLander-v2')

    # TODO: Create the model.
    print("Gamma is: ", gamma)
    print("Test Frequency: ", test_frequency)
    print("num_test_epsiodes: ", num_test_epsiodes)

    policy = Agent(env, args.hidden_units)
    policy.apply(policy.init_weights)
    policy.to(device)

    optimizer = optim.Adam(policy.parameters(), lr=lr)

    episodes, scores, stds = train_agent(policy=policy, env=env, optimizer=optimizer, writer=writer, args=args)
    np.savez('runs/'+'reinforce_reward_data_'+timestr, episodes, scores, stds) 
    # TODO: Train the model using REINFORCE and plot the learning curve.

if __name__ == '__main__':
    main(sys.argv)