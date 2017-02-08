import tensorflow as tf

class LinearQApproximation:
    def __init__(self, n_states, n_options, sess, use_s0=False,
                 opt=tf.train.AdamOptimizer(0.0001)):
        self.n_states = n_states
        self.n_options = n_options
        self.sess = sess
        self.opt = opt
        
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
        
        self.output = tf.matmul(state_enc, W) + b

        self.omega_place = tf.placeholder(tf.int32, shape=())
        omega_reshaped = tf.expand_dims(self.omega_place, 0)
        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(self.output,
                                                           omega_reshaped))

        self.normalized_output = tf.nn.softmax(self.output)
        self.train_op = self.opt.minimize(self.loss)

    def _get_feed_dict(self, omega, sf, s0=None):
        feed_dict = {self.sf_place: sf,
                     self.omega_place: omega}
        if s0 is not None:
            feed_dict[self.s0_place] = s0
        return feed_dict


    def regress(self, omega, sf, s0=None):
        feed_dict = self._get_feed_dict(omega, sf, s0)
        q_loss, q_omega, _ = self.sess.run([self.loss,
                                            self.normalized_output[0],
                                            self.train_op],
                                           feed_dict=feed_dict)

        for _ in xrange(99):
            self.sess.run(self.train_op, feed_dict=feed_dict)

        return q_omega[omega], q_loss

    def q_value(self, omega, sf, s0=None):
        feed_dict = self._get_feed_dict(omega, sf, s0)
        q_omega = self.sess.run(self.normalized_output[0][omega],
                                feed_dict=feed_dict)

        return q_omega
