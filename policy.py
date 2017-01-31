import tensorflow as tf
import numpy as np

# TODO: maybe better change it to tensorflow?
class QLearningPolicy:
    def __init__(self, n_states, n_actions, state_hash=id, terminate_prob=0.05,
                 learning_rate=0.1, discount=0.9, epsilon=0.05):
        self.n_states = n_states
        self.n_actions = n_actions  # assume action no 0 is "finish"
        self.state_hash = state_hash
        self.terminate_prob = terminate_prob
        self.discount = discount
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        self.q_value = np.zeros([n_states, n_actions])

    def process_transition(self, prev_s, action, reward, next_s):
        prev_hash = self.state_hash(prev_s)
        self.q_value[prev_hash, action] =\
            ((1 - self.learning_rate) * self.q_value[prev_hash, action] +
             self.learning_rate * (reward +
                                   self.discount * self.max_action_val(next_s)))

    def max_action_val(self, state):
        state_hash = self.state_hash(state)
        return np.max(self.q_value[state,:])

    def process_trajectory(self, states, actions, rewards):
        for (p_s, a, r, n_s) in zip(states, actions, rewards, states[1:]):
            self.process_transition(p_s, a, r, n_s)
    
    def get_action(self, state):
        if np.random.uniform() < self.epsilon:
            return np.random.randint(self.n_actions)

        return np.argmax(self.q_value[state, :])
