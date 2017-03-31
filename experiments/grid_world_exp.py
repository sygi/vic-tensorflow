from __future__ import print_function

import gym
import vic_envs
import math
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import argparse
import logging
import os.path
import shutil
import time

from policy import QLearningPolicy
from prior import FixedUniformDiscretePrior
from q_approx import LinearQApproximation
from tools import Trajectory, PlotRobot

plt.ion()

class GridWorldExperiment():
    def __init__(self, n_options=10, logger=None, plotting=False,
                 log_tf_graph=False):
        if logger is None:
            logger = logging.getLogger("logger")
            logger.setLevel(logging.INFO)
        self.logger = logger

        self.n_options = n_options
        self.env = gym.make("deterministic-grid-world-v0")
        self.n_actions = self.env.action_space.n
        self.n_states = 1 + reduce(lambda x,y: x*y,
             map(lambda x: x.n, self.env.observation_space.spaces))

        if plotting:
            self.plot_robots = [PlotRobot('dqn loss', 0, log_scale=True),
                                PlotRobot('q loss', 1), PlotRobot('rewards', 2)]
        else:
            self.plot_robots = [None] * 3
        self.plotting = self.plot_robots[2]

        self.colors = list('bgrcmyk') + ['magenta', 'lime', 'gray']
        self.build_graph(log_tf_graph)

    def build_graph(self, log_tf_graph):
        self.sess = tf.Session()
        self.prior = FixedUniformDiscretePrior(self.n_options, self.sess)

        state_first_dim = self.env.observation_space.spaces[0].n

        self.state_hash = lambda x: x[1] * state_first_dim + x[0]

        self.policy = QLearningPolicy(self.n_states, self.n_actions-1,
                                      self.n_options, self.sess,
                                      state_hash=self.state_hash,
                                      plotting=self.plot_robots[0])
        self.q_approx = LinearQApproximation(self.n_states, self.n_options,
                                             self.sess,
                                             plotting=self.plot_robots[1])

        if log_tf_graph:
            logdir = "/data/lisatmp2/sygnowsj/tflogs/gridworld"
            exp_id = 1
            while os.path.exists(logdir + str(exp_id)):
                exp_id += 1
            summary_op = tf.summary.FileWriter(logdir + str(exp_id),
                                               self.sess.graph)
        
        self.sess.run(tf.global_variables_initializer())

    def train(self, n_episodes=1000):
        trajectories = []
        for episode in xrange(n_episodes):
            if episode % 1000 == 999:
                self.logger.info("episode %d", episode)
                self.logger.info("==========")
            else:
                self.logger.debug("episode %d", episode)
                self.logger.debug("==========")
            #self.policy.epsilon = 1.0 - (float(episode) / float(n_episodes))

            action = -1
            states_hist = []
            actions_hist = []
            omega, _ = self.prior.sample_omega()
            p_omegas = self.prior.all_p_omegas()
            self.logger.debug("omega: %d", omega)
            self.logger.debug("epsilon: %f", self.policy.epsilon)
            states_hist, actions_hist = self.rollout(omega)

            self.logger.debug("sf: %s", self.env.state)

            q_omegas = self.q_approx.all_q_values(
                self.state_hash(self.env.state))

            self.logger.debug("q(omega|sf) = %s", q_omegas)
            self.logger.debug("p(omega|s0) = %s", p_omegas)

            assert len(p_omegas) == len(q_omegas)
            rewards = map(lambda (p, q): math.log(q) - math.log(p),
                          zip(p_omegas, q_omegas))

            self.logger.debug("rewards: %s", rewards)
            if self.plotting is not None:
                self.plotting.add(rewards[omega], color=self.colors[omega],
                                  averages=True)
            

            # collecting trajectories
            self.logger.debug("saving trajectory")
            trajectories.append(Trajectory(states=states_hist,
                actions=actions_hist, rewards=rewards))
            self.q_approx.add_to_memory(omega,
                                        self.state_hash(self.env.state))
            if episode % 64 == 63:
                self.logger.debug("regressing q")
                self.q_approx.regress()

                for t in trajectories:
                    p_omegas = self.prior.all_p_omegas()
                    q_omegas = self.q_approx.all_q_values(t.states[-1])
                    t.rewards = map(lambda (p, q): math.log(q) - math.log(p),
                                    zip(p_omegas, q_omegas))
                self.policy.update_policy(trajectories)
                trajectories = []
            # TODO: refactor

            if episode % 400 == 399:
                self.logger.info("episode %d", episode)
                q_app = np.zeros([self.n_states, self.n_options])
                for s in xrange(self.n_states):
                    q_app[s] = self.q_approx.all_q_values(s)
                #    self.logger.info("state %d, q(omega): %s",
                #                     s, q_app[s]))
                self.logger.info("best final state for each option")
                self.logger.info("%s", np.argmax(q_app, axis=0))
                self.logger.info("with value:")
                self.logger.info("%s", np.max(q_app, axis=0))

                self.logger.info("most probable option for each state:")
                self.logger.info("%s", np.argmax(q_app, axis=1))
                self.logger.info("with value:")
                self.logger.info("%s", np.max(q_app, axis=1))


    def rollout(self, omega, render=False):
        self.env.reset()
        action = -1
        states_hist = []
        actions_hist = []
        while not self.policy.is_terminal(action):
            if render:
                self.env._render()
                time.sleep(0.01)
            self.logger.debug("state: %s, state_hash: %d", self.env.state,
                              self.state_hash(self.env.state))
            states_hist.append(self.state_hash(self.env.state))
            # TODO: use state from the env.step

            action = self.policy.get_action(self.state_hash(self.env.state),
                                            omega)
            self.logger.debug("action: %d", action)
            self.env.step(action)
            actions_hist.append(action)

        if render:
            self.env._render(agent_color="BLUE")
            time.sleep(1.)

        return states_hist, actions_hist

if __name__ == "__main__":
    logger = logging.getLogger("mylogger")  # regular logging clashes with gym

    parser = argparse.ArgumentParser()
    parser.add_argument("--log", dest="log", action="store_const",
                        const=True, default=False, help="turn on logging")
    parser.add_argument("--plot", dest="plot", action="store_const",
                        const=True, default=False,
                        help="turn on realtime plotting")
    parser.add_argument("--n_episodes", dest="n_episodes", action="store",
                        default=2000, nargs='?', type=int,
                        help="number of episodes")
    parser.add_argument('--no-roll', dest="no_roll", action="store_const",
                        const=True, default=False,
                        help="disable rollout after training")
    args = parser.parse_args()
    if args.log:
        logger.setLevel(logging.DEBUG)
        hdlr = logging.FileHandler('gridworld.log')
        hdlr.setFormatter(logging.Formatter('D: %(message)s'))
        logger.addHandler(hdlr)
    else:
        logger.setLevel(logging.INFO)

    experiment = GridWorldExperiment(logger=logger, plotting=args.plot)
    logger.info("starting training")

    experiment.train(n_episodes=args.n_episodes)
    logger.info("finished training, starting eval")

    if not args.no_roll:
        samples = 20
        full_reward_sum = 0.
        for omega in xrange(experiment.n_options):
            logger.info("omega %d", omega)
            reward_sum = 0.
            for _ in xrange(samples):
                experiment.rollout(omega)
                q_omega = experiment.q_approx.q_value(
                    omega, experiment.state_hash(experiment.env.state))
                p_omega = experiment.prior.p_omega(omega)
                logger.info("final state %s", experiment.env.state)

                reward_sum += math.log(q_omega) - math.log(p_omega)

            average_reward = reward_sum / samples
            full_reward_sum += average_reward
            logger.info("omega %d average reward %f (log %f)", omega,
                        average_reward, math.exp(average_reward))

        logger.info("reward: %f", full_reward_sum / experiment.n_options)
