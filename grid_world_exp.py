from __future__ import print_function

import gym
import vic_envs
import math
import tensorflow as tf

import argparse
import logging

from policy import QLearningPolicy
from prior import FixedUniformDiscretePrior
from q_approx import LinearQApproximation

logger = logging.getLogger("mylogger")  # regular logging clashes with gym
parser = argparse.ArgumentParser()
parser.add_argument("--log", dest="log", action="store_const",
                    const=True, default=False, help="turn on logging")
args = parser.parse_args()
if args.log:
    logger.setLevel(logging.DEBUG)

n_options = 10
n_episodes = 10000
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
    logger.debug("n_actions: %d, n_states %d" % (n_actions, n_states))
    for episode in xrange(n_episodes):
        if episode % 1000 == 32:
            print("ep:", episode)
        if episode == 9900:
            logger.setLevel(logging.DEBUG)
        logger.debug("episode %d" % episode)
        logger.debug("==========")
        policy.epsilon = 1.0 - (float(episode) / float(n_episodes))
        action = -1
        states_hist = []
        actions_hist = []
        omega, p_omega = prior.sample_omega()
        policy.set_omega(omega)
        while not policy.is_terminal(action):
            logger.debug("omega: %d" % policy.omega)
            logger.debug("state: %s, state_hash: %d" %
                         (env.state, state_hash(env.state)))
            states_hist.append(state_hash(env.state))

            action = policy.get_action(state_hash(env.state))
            logger.debug("action: %d" % action)
            logger.debug("is terminal: %s" % policy.is_terminal(action))
            actions_hist.append(action)
        
        logger.debug("sf: %s" % env.state)
        q_omega = q_approx.regress(omega, state_hash(env.state))
        logger.debug("q(omega|sf) = %f" % q_omega)
        logger.debug("p(omega|s0) = %f" % p_omega)

        rewards = [math.log(q_omega) - math.log(p_omega)] * len(actions_hist)
        logger.debug("reward: %f" % rewards[0])

        policy.process_trajectory(states_hist, actions_hist, rewards)
        logger.debug("policy updated")
