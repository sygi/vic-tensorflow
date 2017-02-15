import tensorflow as tf
import numpy as np

# TODO: maybe better change it to tensorflow?
class QLearningPolicy(object):
    def __init__(self, n_states, n_actions, n_options, state_hash=id,
                 terminate_prob=0.05, learning_rate=0.1, discount=0.9,
                 epsilon=0.1):

        self.n_states = n_states
        self.n_actions = n_actions  # assume action no 0 is "finish"
        self.state_hash = state_hash
        self.terminate_prob = terminate_prob
        self.discount = discount
        self.learning_rate = learning_rate
        self._epsilon = epsilon

        self.min_epsilon = 0.05

        self.q_value = np.random.random([n_options, n_states, n_actions])/1.e6

    def process_transition(self, prev_s, action, reward, next_s):
        self.q_value[self.omega, prev_s, action] =\
            (((1 - self.learning_rate) *
              self.q_value[self.omega, prev_s, action]) +
             (self.learning_rate * 
              (reward + self.discount * self.max_action_val(next_s))))

    def max_action_val(self, state):
        """Assumes state is hashed (an int between 0 and n_states)."""
        return np.max(self.q_value[self.omega, state,:])

    def process_trajectory(self, states, actions, rewards):
        """Assumes states are hashed (ints between 0 and n_states)."""
        for (p_s, a, r, n_s) in zip(states, actions, rewards, states[1:]):
            self.process_transition(p_s, a, r, n_s)

        # last action
        self.q_value[self.omega, states[-1], actions[-1]] =\
            ((1 - self.learning_rate) * self.q_value[self.omega,
                                                     states[-1],
                                                     actions[-1]] + 
             self.learning_rate * rewards[-1])
    
    def get_action(self, state):
        """Assumes state is hashed (an int between 0 and n_states)."""
        assert isinstance(state, int)
        if np.random.uniform() < self.epsilon:
            return np.random.randint(self.n_actions)

        res = np.argmax(self.q_value[self.omega, state, :])
        assert res < self.n_actions
        assert isinstance(res, int)
        return np.argmax(self.q_value[self.omega, state, :])

    def is_terminal(self, action):
        return action == 0

    def set_omega(self, omega):
        self.omega = omega
    
    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        self._epsilon = max(value, self.min_epsilon)
