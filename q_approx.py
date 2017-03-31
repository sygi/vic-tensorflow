import tensorflow as tf

class LinearQApproximation:
    def __init__(self, n_states, n_options, sess, use_s0=False, plotting=None,
                 opt=tf.train.AdamOptimizer(0.001)):
        self.n_states = n_states
        self.n_options = n_options
        self.sess = sess
        self.opt = opt
        self.history_size = 100
        self.experience_history = tf.Variable([[0, 0]] * self.history_size)
        # (option, state)
        self.replay_it = tf.Variable(0)
        # only increasing, the next place to write to will be
        # self.replay_it % self.history_size

        self.plotting = plotting

        self.batch_size = 64
        
        self.build(use_s0)
        

    def build(self, use_s0):
        self.sf_place = tf.placeholder(tf.int32, shape=())
        sf_enc = tf.one_hot(indices=self.sf_place, depth=self.n_states)

        if use_s0:
            self.s0_place = tf.placeholder(tf.int32, shape=())
            s0_enc = tf.one_hot(indices=self.s0_place, depth=self.n_states)
            state_enc = tf.expand_dims(tf.concat_v2([s0_enc, sf_enc], 0), 0)
            input_dim = 2 * self.n_states
        else:
            state_enc = tf.expand_dims(sf_enc, 0)
            input_dim = self.n_states


        W = tf.Variable(tf.truncated_normal([input_dim, self.n_options], 
                                            stddev=0.1))
        b = tf.Variable([0.] * self.n_options)
        
        output = tf.matmul(state_enc, W) + b
        self.normalized_output = tf.nn.softmax(output)

        # extending replay memory
        self.obs_place = tf.placeholder(tf.int32, shape=[2])
        assign_op = self.experience_history[
            self.replay_it % self.history_size].assign(self.obs_place)
        with tf.control_dependencies([assign_op]):
            self.inc_replay_it = self.replay_it.assign_add(1)

        # training part
        # TODO: handle s0

        max_index = tf.minimum(self.history_size, self.replay_it)

        indices = tf.random_uniform([self.batch_size], minval=0,
                                    maxval=max_index, dtype=tf.int32)

        observations = tf.gather(self.experience_history, indices)
        omegas = observations[:, 0]
        final_states = tf.one_hot(observations[:, 1], depth=self.n_states)
        
        assert final_states.get_shape() == (self.batch_size, self.n_states)

        current_output = tf.matmul(final_states, W) + b

        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(current_output,
                                                           omegas))

        normalized_cur_output = tf.nn.softmax(current_output) # batch_size x n_opt

        omegas_reshaped = tf.concat(1,
                                    [tf.expand_dims(tf.range(self.batch_size), 1),
                                     tf.expand_dims(omegas, 1)])

        self.predictions = tf.gather_nd(normalized_cur_output, omegas_reshaped)
        self.train_op = self.opt.minimize(self.loss)

    def _get_feed_dict(self, omega, sf, s0=None):
        feed_dict = {self.obs_place: [omega, sf],
                     self.sf_place: sf}
        if s0 is not None:
            feed_dict[self.s0_place] = s0
        return feed_dict

    
    def add_to_memory(self, omega, sf, s0=None):
        feed_dict = self._get_feed_dict(omega, sf, s0)
        self.sess.run(self.inc_replay_it, feed_dict=feed_dict)

    def regress(self):
        for i in xrange(500):
            loss, _  = self.sess.run([self.loss, self.train_op])
            if i == 0 and self.plotting is not None:
                self.plotting.add(loss)


    def q_value(self, omega, sf, s0=None):
        feed_dict = self._get_feed_dict(omega, sf, s0)
        q_omega = self.sess.run(self.normalized_output[0][omega],
                                feed_dict=feed_dict)

        return q_omega

    def all_q_values(self, sf, s0=None):
        return self.sess.run(self.normalized_output[0], feed_dict={
            self.sf_place: sf})

    # TODO: feeder decorator
