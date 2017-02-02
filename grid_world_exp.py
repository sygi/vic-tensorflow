from __future__ import print_function

import gym
import pdb
import vic_envs
import math
import argparse
import tensorflow as tf

from policy import QLearningPolicy
from prior import FixedUniformDiscretePrior
from q_approx import LinearQApproximation

parser = argparse.ArgumentParser()
parser.add_argument('--debug', dest='debug', action='store_const',
                    const=True, default=False, help='turn on the pdb debugger')
args = parser.parse_args()
if args.debug:
    pdb.set_trace()

n_options = 10
n_episodes = 1000
env = gym.make("grid-world-v0")
n_actions = env.action_space.n
n_states = reduce(lambda x,y: x*y,
                  map(lambda x: x.n, env.observation_space.spaces))

sess = tf.Session()
prior = FixedUniformDiscretePrior(n_options, sess)

state_first_dim = env.observation_space.spaces[0].n
def state_hash(x):
    return x[1] * state_first_dim + x[0]

policy = QLearningPolicy(n_states, n_actions, n_options, state_hash=state_hash)
q_approx = LinearQApproximation(n_states, n_options, sess)

sess.run(tf.global_variables_initializer())

if __name__ == "__main__":
    print("n_actions:", n_actions, "n_states:", n_states)
    for episode in xrange(n_episodes):
        print("\nepisode", episode, "\n==========")
        policy.epsilon = 1.0 - (float(episode) / float(n_episodes))
        action = -1
        states_hist = []
        actions_hist = []
        omega, p_omega = prior.sample_omega()
        policy.set_omega(omega)
        while not policy.is_terminal(action):
            print("omega:", policy.omega)
            print("state:", env.state, "state hash:", state_hash(env.state))
            states_hist.append(state_hash(env.state))

            action = policy.get_action(state_hash(env.state))
            print("action:", action)
            print("is terminal:", policy.is_terminal(action))
            actions_hist.append(action)
        
        print("sf:", env.state)
        q_omega = q_approx.regress(omega, state_hash(env.state))
        print("q(omega|sf) =", q_omega)
        print("p(omega|s0) =", p_omega)

        rewards = [math.log(q_omega) - math.log(p_omega)] * len(actions_hist)
        print("reward:", rewards[0])

        policy.process_trajectory(states_hist, actions_hist, rewards)
        print("policy updated")
