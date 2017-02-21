from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np

class Trajectory():
    def __init__(self, omega, states, actions, rewards):
        self.omega = omega
        self.states = states
        self.actions = actions
        self.rewards = rewards

class PlotRobot():
    def __init__(self, name, figure_id, log_scale=False):
        self.label = name
        self.it = 0
        self.points = []
        self.figure_id = figure_id
        self.log_scale = log_scale

    def add(self, point):  
        self.points.append(point)  # TODO: refactor # TODO: some colors
        self.it += 1
        plt.figure(self.figure_id)
        if self.log_scale:
            plt.yscale('log')
        plt.scatter(self.it, point, label=self.label)

        plt.xlim(0, self.it - self.it%100 + 100)
        plt.legend([self.label])
#        plt.semilogy(self.points, figure=self.fig, label=self.label)
        plt.pause(0.001)
