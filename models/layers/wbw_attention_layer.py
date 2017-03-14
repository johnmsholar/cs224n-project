import sys
sys.path.insert(0, '../')
import tensorflow as tf
from util import multiply_3d_by_2d
class WBWAttentionLayer:
    """Wrapper around attention. Given two texts A and B, represent A's hidden states
       as an attention vector and apply attention to B's output
       Attention derived from Rocktaschel et. al paper 
       (https://arxiv.org/pdf/1509.06664.pdf)
    """
    def __init__(self, hidden_size, A_time_steps):
        self.hidden_size = hidden_size
        self.A_time_steps = A_time_steps

    def __call__(self, Y, b_states, scope=None):
        """
        Args:
            Y: matrix of hidden vectors for A. [batch, A_time_steps, hidden_size]
            b_states: the hidden vectors coming out of B [batch, B_time_steps, hidden_size]
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
            W_r = tf.get_variable("W_r", shape=[self.hidden_size, self.hidden_size])
            W_t = tf.get_variable("W_t", shape=[self.hidden_size, self.hidden_size])

            # initialize r to all 0s
            b_states = tf.transpose(b_states, perm=[1, 0, 2])
            batch_shape = tf.shape(Y)[0]
            initializer = tf.zeros(shape=[batch_shape, self.hidden_size])

            M_component_1 = multiply_3d_by_2d (Y, W_y)

            def perform_attn (prev_r, h_i):
                h_i_plus_w_r = tf.matmul(h_i, W_h) + tf.matmul(prev_r, W_r) # batch x hidden_size
                multiplications = tf.constant([self.A_time_steps, 1])
                h_i_plus_w_r = tf.tile(h_i_plus_w_r, multiplications)
                M_component_2 = tf.reshape(h_i_plus_w_r, shape=[-1, self.A_time_steps, self.hidden_size])
                M = tf.tanh(M_component_1 + M_component_2) #dim: batch x time_steps x hidden_size
                alpha = tf.nn.softmax(multiply_3d_by_2d(M, w)) # dim: batch x time_steps
                Y_t = tf.transpose(Y, perm=[0,2,1]) # dim: batch x hidden_size x A_time_steps
                r= tf.matmul(Y_t, alpha) # dim: batch x hidden x 1
                r = tf.reshape(r, [tf.shape(r)[0], r.get_shape().as_list()[1]]) # collapse. dim: batch x hidden
                r += tf.tanh(tf.matmul(prev_r, W_t)) # dim batch x hidden
                return r

            r = tf.scan(perform_attn, b_states, initializer)
            r_n = r[-1,:,:]
            h_n = b_states[-1,:,:]
            h_star_one = tf.matmul(r_n, W_p)
            h_star_two = tf.matmul(h_n, W_x)
            output = tf.tanh(h_star_one + h_star_two) # dim: batch x hidden
        return output