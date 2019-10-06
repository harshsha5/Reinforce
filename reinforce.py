import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
import gym

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Reinforce(object):
    # Implementation of the policy gradient method REINFORCE.

    def __init__(self, model, lr):
        self.model = model
        model.compile(loss=keras.losses.mean_squared_error,
             optimizer=tf.train.AdamOptimizer(learning_rate=lr),
             metrics=['accuracy'])

        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here.

    def train(self, env, gamma=1.0):
        # Trains the model on a single episode using REINFORCE.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.
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
        states.append(curr_state)
        time_step = 0
        done = False
        #For our case the len(states) = len(actions)+1 = len(rewards)+1 //VERIFY
        while(not done):
            action = self.model.predict(curr_state)
            new_state, reward, done, info = self.env.step(action)
            states.append(new_state)
            rewards.append(reward)
            actions.append(action)
            curr_state = new_state
        return states, actions, rewards


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float, default=5e-4, help="The learning rate.")

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
    model = keras.models.Sequential([
    keras.layers.Dense(16, input_dim=num_states, activation='relu',bias_initializer='zeros',kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform', seed=None)),
    keras.layers.Dense(16, activation='relu',bias_initializer='zeros',kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform', seed=None)),
    keras.layers.Dense(16, activation='relu',bias_initializer='zeros',kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform', seed=None)),
    keras.layers.Dense(num_actions, activation='softmax',bias_initializer='zeros',kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform', seed=None))]) 
    # print(model.summary())
    # config = model.get_config()
    # print(config)
    return model


def main(args):
    # Parse command-line arguments.
    args = parse_arguments()
    num_episodes = args.num_episodes
    lr = args.lr
    render = args.render

    # Create the environment.
    env = gym.make('LunarLander-v2')

    # TODO: Create the model.
    model = make_model(env)
    agent = Reinforce(model,lr)

    # TODO: Train the model using REINFORCE and plot the learning curve.


if __name__ == '__main__':
    main(sys.argv)
