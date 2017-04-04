from __future__ import print_function
import matplotlib.pyplot as plt

class Trajectory():
    def __init__(self, states, actions, rewards):
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

    def add(self, point, color='b', marker='.', averages=False):
        self.points.append(point)  # TODO: refactor # TODO: some colors
        self.it += 1
        plt.figure(self.figure_id)
        if self.log_scale:
            plt.yscale('log')
        plt.scatter(self.it, point, marker=marker, color=color)

        plt.xlim(0, self.it - self.it % 100 + 100)
        if averages and self.it % 200 == 0:
            ave = sum(self.points[-200:])/len(self.points[-200:])
            plt.plot([self.it - 200, self.it], [ave, ave], color='black')

        plt.legend([self.label], loc='upper left')
        plt.pause(0.0001)
