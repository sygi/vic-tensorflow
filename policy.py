import tensorflow as tf
import numpy as np
from random import shuffle

# TODO: maybe better change it to tensorflow?
class QLearningPolicy(object):
    def __init__(self, n_states, n_actions, n_options, sess, state_hash=id,
                 plotting=None, terminate_prob=0.05,
                 opt=tf.train.AdamOptimizer(0.001), discount=0.95,
                 epsilon=0.0, batch_size=32):

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

        self.min_epsilon = 0.0
        self._epsilon = epsilon
        self.build()

    def build(self):
        W = tf.Variable(tf.truncated_normal(
            [self.n_states, self.output_size], stddev=0.1))
        b = tf.Variable([0.] * self.output_size)

        def get_q_value(state_placeholder):
            state_enc = tf.one_hot(indices=state_placeholder, depth=self.n_states)
            return tf.reshape(tf.matmul(state_enc, W) + b,
                              [-1, self.n_actions, self.n_options])

        self.state_place = tf.placeholder(tf.int32, shape=[self.batch_size],
                                          name='state')
        self.sf_place = tf.placeholder(tf.int32, shape=[self.batch_size],
                                       name='final_state')
        self.action_place = tf.placeholder(tf.int32, shape=[self.batch_size],
                                           name='action')
        output_prev = get_q_value(self.state_place)

        print "output prev shape:", output_prev.get_shape()
        indices = tf.stack([tf.range(self.batch_size), self.action_place], 1)
        print "indices shape", indices.get_shape()

        q_value = tf.gather_nd(output_prev, indices)
        print "q value shape:", q_value.get_shape()
        t_assert = tf.assert_equal(q_value.get_shape(),
                                   [self.batch_size, self.n_options])

        output_next = get_q_value(self.sf_place)

        next_state_val = tf.reduce_max(output_next, axis=1)
        # [self.batch_size, self.n_options]

        self.omega_place = tf.placeholder(tf.int32, shape=[self.batch_size],
                                          name='omega')
        self.reward_place = tf.placeholder(tf.float32, shape=[self.batch_size],
                                           name='reward')
        reward_exp = tf.expand_dims(self.reward_place, 1)

        loss = tf.square(
            q_value - (reward_exp + self.discount * next_state_val))

        t_ass2 = tf.assert_equal(loss.get_shape(), [self.batch_size,
                                                 self.n_options])

        indices2 = tf.stack([tf.range(self.batch_size), self.omega_place], 1)

        loss_omega = tf.gather_nd(loss, indices)
        t_ass3 = tf.assert_equal(loss_omega.get_shape(), [self.batch_size])
        self.full_loss = tf.reduce_mean(loss_omega)

        with tf.control_dependencies([t_assert, t_ass2, t_ass3]):
            self.train_op = self.opt.minimize(self.full_loss)

        self.single_state_place = tf.placeholder(tf.int32, shape=(),
                                                 name='single_state')
        self.single_omega_place = tf.placeholder(tf.int32, shape=(),
                                                 name='single_omega')

        # TODO: having two separate sets of placeholders is super shitty
        output_single = get_q_value(tf.expand_dims(self.single_state_place, 0))
        output_perm = tf.transpose(output_single[0], perm=[1, 0])

        self.action = tf.argmax(output_perm[self.single_omega_place], axis=0)

    def process_transitions(self, transitions, plot=False):
        p_s = [t[0] for t in transitions]
        a = [t[1] - 1 for t in transitions]
        r = [t[2] for t in transitions]
        n_s = [t[3] for t in transitions]
        omega = [t[4] for t in transitions]

        loss, _ = self.sess.run([self.full_loss, self.train_op], feed_dict=
                                {self.state_place: p_s,
                                 self.sf_place: n_s,
                                 self.action_place: a,
                                 self.reward_place: r,
                                 self.omega_place: omega})

        if plot and self.plotting is not None:
            self.plotting.add(loss)

    def process_trajectory(self, states, actions, rewards, omega):
        assert all(isinstance(s, int) for s in states)

        for (p_s, a, r, n_s) in zip(states, actions, rewards, states[1:]):
            self.process_transition(p_s, a, r, n_s, omega)

    def update_policy(self, trajectories):
        transitions = []  # TODO: change to at least np.array
        for t in trajectories:
            for (p_s, a, r, n_s) in zip(t.states, t.actions, t.rewards,
                                        t.states[1:]):
                transitions.append((p_s, a, r, n_s, t.omega))

        for j in xrange(50):
            shuffle(transitions)

            for i in xrange(len(transitions)/self.batch_size):
                self.process_transitions(
                    transitions[i*self.batch_size:(i+1)*self.batch_size],
                    i == 0 and j == 0
                )



    def get_action(self, state, omega):
        assert isinstance(state, int)
        if np.random.uniform() < self.terminate_prob:
            return 0  # terminate
        
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
