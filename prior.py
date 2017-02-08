import tensorflow as tf

class FixedUniformDiscretePrior:
    def __init__(self, n_options, sess):
        self.n_options = n_options
        self.sess = sess
        self.picker = tf.multinomial([self.n_options * [1.]], 1)[0][0]

    def p_omega(self, omega):
        return 1./self.n_options

    def sample_omega(self):
        """Returns an id of an option as a tensor of shape [1, 1]."""
        return self.sess.run(self.picker), 1./self.n_options


# TODO: other priors
