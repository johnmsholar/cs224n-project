import sys
sys.path.insert(0, '../')
import tensorflow as tf
from util import multiply_3d_by_2d
class AttentionLayer:
    """Wrapper around attention. Given two texts A and B, represent A's hidden states
       as an attention vector and apply attention to B's output
       Attention derived from Rocktaschel et. al paper 
       (https://arxiv.org/pdf/1509.06664.pdf)
    """
    def __init__(self, hidden_size, A_time_steps):
        self.hidden_size = hidden_size
        self.A_time_steps = A_time_steps

    def __call__(self, Y, h_n, scope=None):
        """
        Args:
            Y: matrix of hidden vectors for A. [batch, A_time_steps, hidden_size]
            h_n: the last hidden vector of the B [batch, hidden_size]
            scope: is the name of the scope to be used when defining the variables inside.
        Returns:
            output: a tensor of size [batch x hidden]
        """
        scope = scope or type(self).__name__

        with tf.variable_scope(scope, initializer=tf.contrib.layers.xavier_initializer()):
            # Weight matrices to train
            W_y = tf.get_variable("W_y", shape=[self.hidden_size, self.hidden_size])
            W_h = tf.get_variable("W_h", shape=[self.hidden_size, self.hidden_size])
            w = tf.get_variable("w", shape=[self.hidden_size, 1])
            W_x = tf.get_variable("W_x", shape=[self.hidden_size, self.hidden_size])
            W_p = tf.get_variable("W_p", shape=[self.hidden_size, self.hidden_size])
            
            M_component_1 = multiply_3d_by_2d (Y, W_y)

            mult_h_n = tf.matmul(h_n, W_h) # batch x hidden_size
            multiplications = tf.constant([self.A_time_steps, 1])
            mult_h_n = tf.tile(mult_h_n, multiplications)
            M_component_2 = tf.reshape(mult_h_n, shape=[-1, self.A_time_steps, self.hidden_size])
            
            M = tf.tanh(M_component_1 + M_component_2) #dim: batch x time_steps x hidden_size
            alpha = tf.nn.softmax(multiply_3d_by_2d(M, w)) # dim: batch x time_steps x 1
            Y_t = tf.transpose(Y, perm=[0,2,1]) # dim: batch x hidden_size x A_time_steps

            r= tf.matmul(Y_t, alpha) # dim: batch x hidden x 1
            r = tf.reshape(r, [tf.shape(r)[0], r.get_shape().as_list()[1]]) # collapse. dim: batch x hidden

            h_star_one = tf.matmul(r, W_p)
            h_star_two = tf.matmul(h_n, W_x)
            output = tf.tanh(h_star_one + h_star_two) # dim: batch x hidden
        return output