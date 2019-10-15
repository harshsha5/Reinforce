import sys
import argparse
import numpy as np
import gym

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from tensorboardX import SummaryWriter
import time

from reinforce import Agent, generate_episode, test_agent

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

def train(env, policy, value_policy, policy_optimizer, value_policy_optimizer, N, gamma):
    states, actions, rewards, log_probs = generate_episode(env, policy, num_ep=1)
    
    V_all = value_policy(torch.from_numpy(np.array(states.squeeze())).float().to(device)).squeeze()
    V_end = V_all[N:]
    V_end = torch.cat((V_end, torch.zeros(N).float().to(device))) * torch.tensor(pow(gamma, N)).float().to(device)

    rewards_tensor = torch.cat((torch.from_numpy(np.array(rewards)/10).float().to(device), torch.zeros(N-1).float().to(device)))
    gamma_multiplier = torch.tensor(np.geomspace(1, pow(gamma, N-1), num=N)).float().to(device)
    
    R_t = []
    for i in range(len(states)):
        R_t.append(V_end[i] + (gamma_multiplier * rewards_tensor[i:i+N]).sum())
    R_t = torch.stack(R_t).float().to(device)

    difference = R_t - V_all
    detached_difference = difference.detach()
    
    L_policy = (detached_difference * -torch.stack(log_probs).squeeze()).sum()
    L_value_policy = torch.pow(difference, 2).mean()
    
    policy_optimizer.zero_grad()
    value_policy_optimizer.zero_grad()
    
    L_policy.backward()
    L_value_policy.backward()

    policy_optimizer.step()
    value_policy_optimizer.step()

    return L_policy.item(), L_value_policy.item()
    # for i in range(len(rewards)-1):


def train_agent(policy, value_policy, env, policy_optimizer, value_policy_optimizer, writer, args, save_path):
    scores = []
    episodes = []
    stds = []
    for e in range(args.num_episodes):
        loss_policy, loss_value = train(env, policy, value_policy, policy_optimizer, value_policy_optimizer, args.n, args.gamma)
        writer.add_scalar("train/Policy Loss", loss_policy, e)
        writer.add_scalar("train/Value Policy Loss", loss_value, e)
        print("Completed episode %d, with Policy loss: %f and Value Policy Loss: %f"%(e, loss_policy, loss_value))
        if(e % args.test_frequency==0):
            score, std = test_agent(env, policy, args.num_test_epsiodes)
            writer.add_scalar('test/Reward', score , e)
            scores.append(score)
            episodes.append(e)
            stds.append(std)
            np.savez(save_path+'reward_data', episodes, scores, stds) 
        if(e % args.save_model_frequency == 0):
            torch.save({
                'epoch': e,
                'policy_state_dict': policy.state_dict(),
                'value_policy_state_dict': value_policy.state_dict(),
                'policy_optimizer_state_dict': policy_optimizer.state_dict(),
                'value_policy_optimizer_state_dict': value_policy_optimizer.state_dict(),
                'loss_policy': loss_policy,
                'loss_value': loss_value
                }, save_path+'checkpoint'+str(e)+'.pth')
    return episodes, scores, stds

def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The actor's learning rate.")
    parser.add_argument('--critic-lr', dest='critic_lr', type=float,
                        default=1e-4, help="The critic's learning rate.")
    parser.add_argument('--gamma', dest='gamma', type=float,
                        default=0.99, help="Gamma value.")
    parser.add_argument('--n', dest='n', type=int,
                        default=20, help="The value of N in N-step A2C.")
    parser.add_argument('--test_frequency', dest='test_frequency', type=int, default=200, help="After how many policies do I test the model")
    parser.add_argument('--num_test_epsiodes', dest='num_test_epsiodes', type=int, default=100, help="For how many policies do I test the model")
    parser.add_argument('--hidden_units', dest='hidden_units', type=int, default=16, help="Number of Hidden units in the linear layer")
    parser.add_argument('--save_model_frequency', dest='save_model_frequency', type=int, default=1000, help="Frequency of saving the model")
    parser.add_argument('--load_model', dest='load_model', type=str, default="", help="load path of the model")

    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    return parser.parse_args()


def main(args):
    # Parse command-line arguments.
    args = parse_arguments()
    num_episodes = args.num_episodes
    lr = args.lr
    critic_lr = args.critic_lr
    n = args.n
    render = args.render
    test_frequency = args.test_frequency
    num_test_epsiodes = args.num_test_epsiodes

    timestr = time.strftime("%Y%m%d-%H%M%S")
    save_path = "runs/"+str(args.n)+"a2c_"+timestr+'/'

    # Create the environment.
    env = gym.make('LunarLander-v2')

    # TODO: Create the model.
    policy = Agent(env, args.hidden_units)
    policy.to(device)
    policy_optimizer = optim.Adam(policy.parameters(), lr=lr)

    value_policy = Agent(env, args.hidden_units, 1)
    value_policy.to(device)
    value_policy_optimizer = optim.Adam(value_policy.parameters(), lr=critic_lr)

    if(args.load_model == ""):
        writer = SummaryWriter(save_path)
        policy.apply(policy.init_weights)
        value_policy.apply(value_policy.init_weights)
        episodes, scores, stds = train_agent(policy=policy, value_policy=value_policy, env=env, policy_optimizer=policy_optimizer, value_policy_optimizer=value_policy_optimizer, writer=writer, args=args, save_path=save_path)
    else:
        checkpoint = torch.load(args.load_model+'.pth')
        policy.load_state_dict(checkpoint['model_state_dict'])
        value_policy.load_state_dict(checkpoint['model_state_dict'])
        policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        value_policy_optimizer.load_state_dict(checkpoint['value_policy_optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        env = gym.wrappers.Monitor(env, "video.mp4", force=True)
        generate_episode(env, policy, True)
    # TODO: Train the model using A2C and plot the learning curves.


if __name__ == '__main__':
    main(sys.argv)
