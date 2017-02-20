import tensorflow as tf
import numpy as np

# TODO: maybe better change it to tensorflow?
class QLearningPolicy(object):
    def __init__(self, n_states, n_actions, n_options, sess, state_hash=id,
                 terminate_prob=0.05, opt=tf.train.AdamOptimizer(0.0002),
                 discount=0.9, epsilon=0.0):

        self.n_states = n_states
        self.n_actions = n_actions  # assume action no 0 is "finish"
        self.n_options = n_options
        self.output_size = self.n_actions * self.n_options
        self.sess = sess
        self.state_hash = state_hash
        self.terminate_prob = terminate_prob
        self.discount = discount
        self.opt = opt

        self.min_epsilon = 0.0
        self._epsilon = epsilon
        self.build()

    def build(self):
        W = tf.Variable(tf.truncated_normal(
            [self.n_states, self.output_size], stddev=0.1))
        b = tf.Variable([0.] * self.output_size)

        self.state_place = tf.placeholder(tf.int32, shape=(), name='state')
        self.action_place = tf.placeholder(tf.int32, shape=(), name='action')

        state_enc = tf.expand_dims(tf.one_hot(indices=self.state_place,
                                              depth=self.n_states), 0)
        output_prev = tf.reshape(tf.matmul(state_enc, W) + b,
                                 [self.n_options, self.n_actions])
        q_val_prev = output_prev[:, self.action_place]  # [self.n_options]

        self.sf_place = tf.placeholder(tf.int32, shape=(), name='final_state')
        sf_enc = tf.expand_dims(tf.one_hot(indices=self.sf_place,
                                           depth=self.n_states), 0)
        output_next = tf.reshape(tf.matmul(sf_enc, W) + b,
                                 [self.n_options, self.n_actions])
        # TODO: refactor

        next_state_val = tf.reduce_max(output_next, axis=1)  # [self.n_options]

        self.omega_place = tf.placeholder(tf.int32, shape=(), name='omega')
        self.reward_place = tf.placeholder(tf.float32, shape=(), name='reward')

        loss = tf.square(q_val_prev
                         - (self.reward_place + self.discount * next_state_val))

        ass = tf.assert_equal(loss.get_shape(), [self.n_options])
        self.train_op = self.opt.minimize(loss[self.omega_place])

        self.action = tf.argmax(output_prev[self.omega_place], axis=0)

    def process_transition(self, prev_s, action, reward, next_s, omega):
        self.sess.run(self.train_op, feed_dict={self.state_place: prev_s,
                                                self.sf_place: next_s,
                                                self.action_place: action - 1,
                                                self.reward_place: reward,
                                                self.omega_place: omega})

    def process_trajectory(self, states, actions, rewards, omega):
        assert all(isinstance(s, int) for s in states)

        for (p_s, a, r, n_s) in zip(states, actions, rewards, states[1:]):
            self.process_transition(p_s, a, r, n_s, omega)

    def update_policy(self, trajectories):
        for _ in xrange(10):
            for t in trajectories:
                self.process_trajectory(t.states, t.actions, t.rewards, t.omega)


    def get_action(self, state, omega):
        assert isinstance(state, int)
        if np.random.uniform() < self.terminate_prob:
            return 0  # terminate
        
        if np.random.uniform() < self.epsilon:
            return 1 + np.random.randint(self.n_actions)

        res = self.sess.run(self.action, feed_dict={self.state_place: state,
                                                    self.omega_place: omega})
        assert res < self.n_actions
        return 1 + res

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
