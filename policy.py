import tensorflow as tf
import numpy as np
from random import shuffle

# TODO: maybe better change it to tensorflow?
class QLearningPolicy(object):
    def __init__(self, n_states, n_actions, n_options, sess, state_hash=id,
                 plotting=None, terminate_prob=0.05,
                 opt=tf.train.AdamOptimizer(0.0001), discount=0.99,
                 epsilon=0.20, batch_size=32):

        self.n_states = n_states
        self.n_actions = n_actions  # assume action no 0 is "finish"
        self.n_options = n_options
        self.output_size = self.n_actions * self.n_options
        self.sess = sess
        self.state_hash = state_hash
        self.terminate_prob = terminate_prob
        self.plotting = plotting
        self.discount = discount
        self.opt = opt
        self.batch_size = batch_size
        self.action_it = 0

        self.min_epsilon = 0.001  # change from 0.001 here
        self._epsilon = epsilon
        self.build()

    def build(self):
        W = tf.Variable(tf.truncated_normal(
            [self.n_states, self.output_size], stddev=1.))  # + 1
        b = tf.Variable([0.] * self.output_size)

        def get_q_value(state_placeholder):
            state_enc = tf.one_hot(indices=state_placeholder,
                                   depth=self.n_states)  # + 1
            return tf.reshape(tf.matmul(state_enc, W) + b,
                              [-1, self.n_actions, self.n_options])

        self.state_place = tf.placeholder(tf.int32, shape=[self.batch_size],
                                          name='state')
        self.sf_place = tf.placeholder(tf.int32, shape=[self.batch_size],
                                       name='final_state')
        self.action_place = tf.placeholder(tf.int32, shape=[self.batch_size],
                                           name='action')
        output_prev = get_q_value(self.state_place)

        indices = tf.stack([tf.range(self.batch_size), self.action_place], 1)

        q_value = tf.gather_nd(output_prev, indices)
        t_ass = tf.assert_equal(q_value.get_shape(),
                                [self.batch_size, self.n_options])

        output_next = get_q_value(self.sf_place)

        next_state_val = tf.reduce_max(output_next, axis=1)
        t_ass4 = tf.assert_equal(next_state_val.get_shape(),
                                 [self.batch_size, self.n_options])

        self.reward_place = tf.placeholder(
            tf.float32, shape=[self.batch_size, self.n_options], name='rewards')

        loss = tf.square(
            q_value - (self.reward_place + self.discount * next_state_val))

        t_ass2 = tf.assert_equal(loss.get_shape(), [self.batch_size,
                                                    self.n_options])

        self.loss_red = tf.reduce_mean(loss)

        with tf.control_dependencies([t_ass, t_ass2, t_ass4]):
            self.train_op = self.opt.minimize(self.loss_red)

        self.single_state_place = tf.placeholder(tf.int32, shape=(),
                                                 name='single_state')
        self.single_omega_place = tf.placeholder(tf.int32, shape=(),
                                                 name='single_omega')

        # TODO: having two separate sets of placeholders is super shitty
        output_single = get_q_value(tf.expand_dims(self.single_state_place, 0))
        output_perm = tf.transpose(output_single[0], perm=[1, 0])

        t_ass3 = tf.assert_equal(output_perm.get_shape(),
                                 [self.n_options, self.n_actions])

        with tf.control_dependencies([t_ass3]):
            self.action = tf.argmax(output_perm[self.single_omega_place], axis=0)

    def process_transitions(self, transitions, plot=False):
        p_s, a, r, n_s = zip(*transitions)
        a = [action - 1 for action in a]

        loss, _ = self.sess.run([self.loss_red, self.train_op],
                                feed_dict={self.state_place: p_s,
                                           self.sf_place: n_s,
                                           self.action_place: a,
                                           self.reward_place: r})

        if plot and self.plotting is not None:
            self.plotting.add(loss)

    def update_policy(self, trajectories):
        transitions = []  # TODO: change to at least np.array
        for t in trajectories:
            for (p_s, a, n_s) in zip(t.states, t.actions, t.states[1:]):
                transitions.append((p_s, a, t.rewards, n_s))

#            transitions.append(
#                (t.states[-1], 1, t.rewards[-1], self.n_states, t.omega))

        for j in xrange(10):
            shuffle(transitions)

            for i in xrange(len(transitions)/self.batch_size):
                self.process_transitions(
                    transitions[i*self.batch_size:(i+1)*self.batch_size],
                    i == 0 and j % 10 == 0)

    def reset_action_it(self):
        self.action_it = 0

    def get_action(self, state, omega):
        assert isinstance(state, int)
        if self.action_it > 5 and np.random.uniform() < self.terminate_prob:
            self.reset_action_it()
            return 0  # terminate

        self.action_it += 1

        if np.random.uniform() < self.epsilon:
            return 1 + np.random.randint(self.n_actions)

        res = self.sess.run(self.action,
                            feed_dict={self.single_state_place: state,
                                       self.single_omega_place: omega})
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
