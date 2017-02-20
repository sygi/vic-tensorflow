from __future__ import print_function

class Trajectory():
    def __init__(self, omega, states, actions, rewards):
        self.omega = omega
        self.states = states
        self.actions = actions
        self.rewards = rewards
