import tensorflow as tf

class LinearQApproximation:
    def __init__(self, n_states, n_options, sess, use_s0=False,
                 opt=tf.train.AdamOptimizer):
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
            state_enc = tf.concat_v2([s0_enc, sf_enc], 0)
            input_dim = 2 * self.n_states
        else:
            state_enc = sf_enc
            input_dim = self.n_states

        W = tf.Variable(tf.truncated_normal([input_dim, self.n_options], 
                                            stddev=0.1))
        b = tf.Variable(tf.constant(0., shape=(self.n_options)))
        
        self.output = tf.matmul(state_enc, W) + b

        self.omega_place = tf.placeholder(tf.int32, shape=())
        output_reshaped = tf.expand_dims(self.output, 0)
        self.loss = tf.reduce_mean(tf.nn_softmax_cross_entropy_with_logits(
            tf.expand_dims(ouptut_reshaped, 0),
            tf.expand_dims(self.omega_place, 0)))


    def regress(self, omega, sf, s0=None):
        feed_dict = {self.sf_place: sf,
                     self.omega_place: omega}
        if s0 is not None:
            feed_dict[self.s0_place] = s0
            
        q_omega, _ = sess.run([self.output, self.opt.minimize(self.loss)],
                              feed_dict=feed_dict)
        
        return q_omega[omega]

