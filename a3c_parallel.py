import collections
import math
import os
import sys
import argparse

import cv2
import numpy as np
import gym

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
# torch.multiprocessing.set_start_method('spawn', force=True)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.distributions import Categorical

from tensorboardX import SummaryWriter
import time


# use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0')
torch.cuda.empty_cache()

class SharedAdamW(optim.AdamW):
    """Implements Adam algorithm with shared states.
    """

    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0):
        super(SharedAdamW, self).__init__(params, lr, betas, eps, weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step'].item()
                bias_correction2 = 1 - beta2 ** state['step'].item()
                step_size = group['lr'] * math.sqrt(
                    bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


class Agent_Breakout(torch.nn.Module):
    def __init__(self, env):
        super(Agent_Breakout, self).__init__()
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.n

        self.conv1 = torch.nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)

        self.fc = torch.nn.Linear(64 * 8 * 8, 256)
        self.fc_out = nn.Linear(256, self.num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc(x))
        x = F.softmax(self.fc_out(x), dim=1)
        return x

    def init_weights(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            m.weight.data.fill_(0)
            m.bias.data.fill_(0)


class CriticAgent_Breakout(torch.nn.Module):
    def __init__(self, env, output=0):
        super(CriticAgent_Breakout, self).__init__()

        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.n if output==0 else output

        self.conv1 = torch.nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)

        self.fc = torch.nn.Linear(64 * 8 * 8, 256)
        self.fc_out = nn.Linear(256, self.num_actions)

    def forward(self, x):
        # print(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc(x))
        x = self.fc_out(x)
        return x

    def init_weights(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            m.weight.data.fill_(0)
            m.bias.data.fill_(0)


def atari_transform(curr_state):
    curr_state = cv2.cvtColor(curr_state, cv2.COLOR_BGR2GRAY)
    curr_state = curr_state[34:34+160, :]
    curr_state = cv2.resize(curr_state, (int(curr_state.shape[1]*0.5), int(curr_state.shape[0]*0.5)))
    curr_state = curr_state/ 255.0
    # cv2.imshow('y', curr_state)
    # cv2.waitKey(0)
    return curr_state


def select_action(env, probs, type="prob_sample"):
    if(type=="prob_sample"):
        m = Categorical(probs)
        action = m.sample()
        return action, m.log_prob(action)
    elif(type=="best"):
        return torch.argmax(probs,axis=1)
    elif(type=="random"):
        return env.action_space.sample()

def generate_episode(env, policy, counter, lock, render=False):
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
    count = 0
    frames_4 = collections.deque([np.zeros(shape=(80, 80))]*3, maxlen=4)
    while(not done):
        if(render):
            env.render()
        if(len(curr_state.shape) == 1):
            curr_state = np.expand_dims(curr_state, 0)
        if(len(curr_state.shape) == 3):
            curr_state = atari_transform(curr_state)
        frames_4.appendleft(curr_state)
        prob = policy(torch.from_numpy(np.asarray(frames_4)).float().unsqueeze(0).to(device))
        action, log_prob = select_action(env, prob)         #VERIFY IF BEST ACTION NEEDS TO BE TAKEN OR RANDOM
        new_state, reward, done, info = env.step(action.item())


        # with lock:
        #     counter.value += 1

        states.append(frames_4)
        actions.append(action)
        rewards.append(reward)
        log_probs.append(log_prob)
        curr_state = new_state
        count += 1
    states = torch.from_numpy(np.array(states))
    return states, actions, rewards, log_probs, count


def test_agent(env, policy, num_test_episodes, counter, lock):
    rewards=[]
    policy.eval()
    for e in range(num_test_episodes):
        _, _, episode_rewards, _, count = generate_episode(env, policy, counter, lock)
        rewards.append(sum(episode_rewards))
    rewards = np.array(rewards)
    policy.train()
    print("Testing - %d episodes mean %f & std deviation %f" %(num_test_episodes, rewards.mean(), rewards.std()))
    return rewards.mean(), rewards.std()


def train( shared_policy, shared_value_policy, policy_optimizer, value_policy_optimizer, N, gamma, args, e, q, counter, lock):

    torch.manual_seed(args.seed + e)

    # Create the environment.
    env = gym.make('Breakout-v0')
    env.seed(args.seed + e)

    policy = Agent_Breakout(env)
    policy.to(device)

    value_policy = CriticAgent_Breakout(env, 1)
    value_policy.to(device)

    if policy_optimizer is None:
        policy_optimizer = optim.AdamW(policy.parameters(), lr=args.lr)
    if value_policy_optimizer is None:
        value_policy_optimizer = optim.AdamW(value_policy.parameters(), lr=args.critic_lr)

    policy.load_state_dict(shared_policy.state_dict())
    value_policy.load_state_dict(shared_value_policy.state_dict())

    states, actions, rewards, log_probs, count = generate_episode(env, policy, counter, lock)

    # import pdb; pdb.set_trace()
    V_all = value_policy(torch.from_numpy(np.array(states.squeeze(dim=2))).float().to(device)).squeeze()
    V_end = V_all[N:]
    V_end = torch.cat((V_end, torch.zeros(N).float().to(device))) * torch.tensor(pow(gamma, N)).float().to(device)

    rewards_tensor = torch.cat((torch.from_numpy(np.array(rewards)/10).float().to(device), torch.zeros(N-1).float().to(device)))
    gamma_multiplier = torch.tensor(np.geomspace(1, pow(gamma, N-1), num=N)).float().to(device)

    R_t = []
    for i in range(len(states)):
        R_t.append(V_end[i] + (gamma_multiplier * rewards_tensor[i:i+N]).sum())
    R_t = torch.stack(R_t).float().to(device)

    difference = R_t.detach() - V_all
    detached_difference = difference.detach()

    L_policy = (detached_difference * -torch.stack(log_probs).squeeze()).sum()
    L_value_policy = torch.pow(difference, 2).mean()

    policy_optimizer.zero_grad()
    value_policy_optimizer.zero_grad()

    L_policy.backward()
    L_value_policy.backward()

    ensure_shared_grads(policy, shared_policy)
    ensure_shared_grads(value_policy, shared_value_policy)

    policy_optimizer.step()
    value_policy_optimizer.step()

    q.put( (L_policy.item(), L_value_policy.item(), e, count) )

    return L_policy.item(), L_value_policy.item(), count
    # for i in range(len(rewards)-1):

def train_agent(policy, value_policy, env, policy_optimizer, value_policy_optimizer, scheduler_policy, scheduler_value_policy, writer, args, save_path, ctx):

    q = ctx.Queue()
    e = ctx.Event()

    scores = []
    episodes = []
    stds = []

    # A3C multi processing
    counter = mp.Value('i', 0)
    lock = mp.Lock()
    processes = []

    e = 0
    # for e in range(args.num_episodes):

    running_loss_policy = 0
    running_loss_value_policy = 0
    avg_loss_policy = 0
    avg_loss_value_policy = 0

    while e < args.num_episodes:

        for i in range(args.num_processes):
            p = ctx.Process(target=train, args=(policy, value_policy, policy_optimizer, value_policy_optimizer, args.n, args.gamma, args, e, q, counter, lock))
            p.start()
            processes.append(p)
            e += 1
        for p in processes:
            p.join()

        for i in range(args.num_processes):
            loss_policy, loss_value, e_, count = q.get()
            running_loss_policy += abs(loss_policy)
            running_loss_value_policy += abs(loss_value)

        writer.add_scalar("train/Policy Loss", loss_policy, e_)
        writer.add_scalar("train/Value Policy Loss", loss_value, e_)
        writer.add_scalar("train/abs_Policy Loss", abs(loss_policy), e)
        writer.add_scalar("train/abs_Value Policy Loss", abs(loss_value), e)
        # print("Completed episode %d of steps %d, with Policy loss: %f and Value Policy Loss: %f"%(e_, count, loss_policy, loss_value))

        # loss_policy, loss_value, count = train(env, policy, value_policy, policy_optimizer, value_policy_optimizer, args.n, args.gamma, e, args, counter, lock)
        # writer.add_scalar("train/Policy Loss", loss_policy, e)
        # writer.add_scalar("train/Value Policy Loss", loss_value, e)
        # # print("Completed episode %d of steps %d, with Policy loss: %f and Value Policy Loss: %f"%(e, count, loss_policy, loss_value))

        if(e % args.test_frequency-args.num_processes==0):
            score, std = test_agent(env, policy, args.num_test_epsiodes, counter, lock)
            scores.append(score)
            episodes.append(e)
            stds.append(std)
            np.savez(save_path+'reward_data', episodes, scores, stds)

            avg_loss_policy = running_loss_policy/args.test_frequency
            avg_loss_value_policy = running_loss_value_policy/args.test_frequency

            running_loss_policy = 0
            running_loss_value_policy = 0


            writer.add_scalar('test/Reward', score , e)
            writer.add_scalar("train/LR_policy", policy_optimizer.param_groups[-1]['lr'], e)
            writer.add_scalar("train/LR_value_policy", value_policy_optimizer.param_groups[-1]['lr'], e)

            if e!=args.num_processes:
                writer.add_scalar("train/Avg_Policy Loss", avg_loss_policy, e)
                writer.add_scalar("train/Avg_Value Policy Loss", avg_loss_value_policy, e)

                scheduler_policy.step(avg_loss_policy)
                scheduler_value_policy.step(avg_loss_value_policy)

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
    parser.add_argument('--num-episodes', dest='num_episodes', type=int, default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float, default=5e-4, help="The actor's learning rate.")
    parser.add_argument('--critic-lr', dest='critic_lr', type=float, default=1e-4, help="The critic's learning rate.")
    parser.add_argument('--gamma', dest='gamma', type=float, default=0.99, help="Gamma value.")
    parser.add_argument('--n', dest='n', type=int, default=20, help="The value of N in N-step A2C.")
    parser.add_argument('--test_frequency', dest='test_frequency', type=int, default=200, help="After how many policies do I test the model")
    parser.add_argument('--num_test_epsiodes', dest='num_test_epsiodes', type=int, default=100, help="For how many policies do I test the model")
    parser.add_argument('--hidden_units', dest='hidden_units', type=int, default=16, help="Number of Hidden units in the linear layer")
    parser.add_argument('--save_model_frequency', dest='save_model_frequency', type=int, default=1000, help="Frequency of saving the model")
    parser.add_argument('--load_model', dest='load_model', type=str, default="", help="load path of the model")

    # A3C Args
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--num-processes', type=int, default=5, help='how many training processes to use (default: 5)')
    parser.add_argument('--no-shared', default=False, help='use an optimizer without shared momentum.')


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
    save_path = "runs/"+"a3c_parallel_"+str(args.num_processes)+'_'+str(args.n)+'_'+str(args.lr)+'_'+str(args.critic_lr)+'_'+timestr+'/'

     # A3C

    # Create the environment.
    env = gym.make('Breakout-v0')

    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    ctx = mp.get_context("spawn")

    torch.manual_seed(args.seed)

    policy = Agent_Breakout(env)
    policy.to(device)

    value_policy = CriticAgent_Breakout(env, 1)
    value_policy.to(device)

    # A3C
    policy.share_memory()
    value_policy.share_memory()

    if args.no_shared:
        policy_optimizer = optim.AdamW(policy.parameters(), lr=lr)
        value_policy_optimizer = optim.AdamW(value_policy.parameters(), lr=critic_lr)
    else:
        policy_optimizer = SharedAdamW(policy.parameters(), lr=lr)
        value_policy_optimizer = SharedAdamW(value_policy.parameters(), lr=critic_lr)

        policy_optimizer.share_memory()
        value_policy_optimizer.share_memory()

    scheduler_policy = optim.lr_scheduler.ReduceLROnPlateau(policy_optimizer, 'min', factor=0.5, patience=3)
    scheduler_value_policy = optim.lr_scheduler.ReduceLROnPlateau(value_policy_optimizer, 'min', factor=0.5, patience=3)

    if(args.load_model == ""):
        writer = SummaryWriter(save_path)
        policy.apply(policy.init_weights)
        value_policy.apply(value_policy.init_weights)
        episodes, scores, stds = train_agent(policy=policy, value_policy=value_policy, env=env, policy_optimizer=policy_optimizer, 
                                             value_policy_optimizer=value_policy_optimizer, scheduler_policy=scheduler_policy, 
                                             scheduler_value_policy=scheduler_value_policy, writer=writer, args=args, save_path=save_path, ctx=ctx)
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
