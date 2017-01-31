import tensorflow as tf

class FixedUniformDiscretePrior:
    def __init__(self, n_options=30):
        self.n_options = n_options

    def sample_omega(self):
        """Returns an id of an option as a tensor of shape [1, 1]."""
        return tf.multinomial([self.n_options * [1.]], 1)

# TODO: other priors
