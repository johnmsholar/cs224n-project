import sys
sys.path.insert(0, '../')

import numpy as np
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

    def compute_attention(self, Y, h_n, W_y, W_h, w, W_x, W_p):
        """ Compute attention.
        """
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
            output = self.compute_attention(Y, h_n, W_y, W_h, w, W_x, W_p)

        return output

def test():
    batch_size = 3
    A_time_steps = 2
    hidden_size = 4
    attention_layer = AttentionLayer(hidden_size, A_time_steps)
    with tf.Session() as session:
        Y = tf.constant([
            [[1, 2, 3, 4],
            [1, 2, 3, 4]],
            [[2, 3, 4, 5],
            [2, 3, 4, 5]],
            [[3, 4, 5, 6],
            [3, 4, 5, 6]]], dtype=tf.float32) # batch_size x A_time_steps x hidden_size
        h_n = tf.constant([
            [1, 2, 89, 12],
            [0, 21, 9, 1],
            [3, 4, 5, 6]], dtype=tf.float32) # batch x hidden_size
        W_y = tf.constant([
            [0, 2, 3, 8],
            [1, 7, 2, 1],
            [1, 1, 1, 1],
            [0, 0, 0, 0]], dtype=tf.float32) # hidden_size x hidden_size
        W_h = tf.constant([
            [0, 0, 0, 0],
            [0, 2, 3, 8],
            [1, 1, 1, 1],
            [1, 7, 2, 1]], dtype=tf.float32) # hidden_size x hidden_size 
        w = tf.constant([[6], [7], [8], [9]], dtype=tf.float32)
        W_x = tf.constant([
            [1, 1, 1, 1],
            [0, 2, 3, 8],
            [1, 7, 2, 1],
            [0, 0, 0, 0]], dtype=tf.float32) # hidden_size x hidden_size
        W_p = tf.constant([
            [0, 2, 3, 8],
            [0, 0, 0, 0],
            [1, 7, 2, 1],
            [1, 1, 1, 1]], dtype=tf.float32) # hidden_size x hidden_size
        result = attention_layer.compute_attention(Y, h_n, W_y, W_h, w, W_x, W_p).eval()

    output = np.array([
        [ 1.,  1.,  1.,  1.],
        [ 1.,  1.,  1.,  1.],
        [ 1.,  1.,  1.,  1.]]
    )

    np.testing.assert_array_equal(result, output)


def main():
    test()

if __name__ == '__main__':
    main()

