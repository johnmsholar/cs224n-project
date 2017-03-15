from attention_base_class import Attention_Base_Class
import sys

sys.path.insert(0, '../')
import tensorflow as tf
from util import multiply_3d_by_2d


class Full_Matching_Attention_Layer(Attention_Base_Class):
    """
    """
    def compute_attention(self, a, b, W):
        """
        Args:
            a: tensor [batch, A_time_steps, hidden_size]
            b: tensor [batch, B_time_steps, hidden_size]
            W: weight matrix [hidden_size x num_perspectives]
        Returns:
            output: tensor [batch x A_time_steps x num_perspectives]
        """
        # a_perm is now [A_time_steps, batch, hidden_size]
        a_perm = tf.transpose(a, perm=[1, 0, 2])
        a_time_steps = a_perm.get_shape()[0]

        # h_n is [batch x hidden_size]
        h_n =  b[:, -1, :]

        idx = tf.constant(0) # current time step index
        cond = lambda i, result: i < a_time_steps
        def body(i, result):
            h_i = a_perm[i, :, :] # 1 x batch x hidden_size
            m_i = score(h_i, h_n, W) # 1 x batch x perspectives (score returns 3D)
            result = tf.concat([result, m_i], axis=0) # building batch x time_steps x persepctive
            return [i+1, result]

        batch_size = tf.shape(a_perm)[1]
        result = tf.zeros(shape=[1, batch_size, self.num_perspectives])
        tf.while_loop(cond, body, [idx, result])
        result = result[1:, :, :]    

