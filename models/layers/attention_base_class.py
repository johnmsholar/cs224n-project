import sys
sys.path.insert(0, '../')
import tensorflow as tf
from util import multiply_3d_by_2d

class Attention_Base_Class(object):
    """
    """
    def __init__(self, num_perspectives):
        self.num_perspectives = num_perspectives

    def compute_score(self, v1, v2, W):
        """
        Args:
            v1: 1 xx_ batch_size x hidden_size
            v2: 1 x batch_size x hidden_size
            W: Scoring Weight Matrix [hidden_size x num_perspectives]
        Returns:
            score:
        """
        hidden_size = v1.get_shape()[2]
        assert hidden_size == v2.get_shape()[2]
        v1_collapse = tf.reshape(v1, shape=[hidden_size, -1]) # hidden x batch
        v2_collapse = tf.reshape(v2, shape=[hidden_size, -1]) # hidden x batch


        idx = tf.constant(0)
        cond = lambda i, result1, result2: i<self.config.num_perspectives
        def body(i, result1, result2):
            W_i = W[:, i] # hidden x 1
            v_1_w_i = tf.multiply(v1_collapse, W_i) # hidden x batch
            v_2_w_i = tf.multiply(v2_collapse, W_i) # hidden x batch
            cos_sim = tf.constant(cosine_similarity(v_1_w_i, v_2_w_i)) # scalar

    def compute_attention(self, a, b, W):
        """
        Args:
            a: tensor [batch, A_time_steps, hidden_size]
            b: tensor [batch, B_time_steps, hidden_size]
            W: weight matrix [hidden_size x num_perspectives]
        Returns:
            output: tensor [batch, A_time_steps, num_perspectives]
        """
        raise NotImplementedError("Each Attention Model must re-implement this method.")

    def __call__(self, A, B, scope=None):
        """
        Args:
            A: tuple of matrices(fw, bw) each one of which is [batch, A_time_steps, hidden_size]
            B: tuple of matrices(fw, bw) each one of which is [batch, B_time_steps, hidden_size]
            scope: is the name of the scope to be used when defining the variables inside.
        Returns:
            output: a tensor of size [batch x hidden]
        """
        scope = scope or type(self).__name__

        with tf.variable_scope(scope):
            # Expand the tuples
            A_fw, A_bw = A[0], A[1]
            B_fw, B_bw = B[0], B[1]

            # Infer hidden size
            hidden_size = A_fw.get_shape()[2]
            assert hidden_size == A_bw.get_shape()[2]
            assert hidden_size == B_fw.get_shape()[2]
            assert hidden_size == B_bw.get_shape()[2]

            # Create Scoring Matrices
            self.W1 = tf.get_variable("W1", shape=[hidden_size, self.num_perspectives])
            self.W2 = tf.get_variable("W2", shape=[hidden_size, self.num_perspectives])

            fw_attention = compute_attention(A_fw, B_fw)
            bw_attention = compute_attention(A_bw, B_bw)
            output = tf.concat([fw_attention, bw_attention], 2)

        return output