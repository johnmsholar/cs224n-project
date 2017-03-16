from attention_base_class import Attention_Base_Class, numpy_reference_compute_score, numpy_generate_A_B_w
import sys
import numpy as np

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

        # h_n is [1 x batch x hidden_size]
        h_n =  tf.transpose(tf.expand_dims(b[:, -1, :], axis=0), perm=[1, 0, 2])

        idx = tf.constant(0) # current time step index
        cond = lambda i, result: i < a_time_steps
        def body(i, result):
            h_i = tf.expand_dims(a_perm[i, :, :], axis=0) # 1 x batch x hidden_size
            m_i = tf.expand_dims(self.compute_score(h_i, h_n, W), axis=0) # 1 x batch x perspectives (score returns 3D)
            result = tf.concat([result, m_i], axis=0) # building time_steps x batch x persepctive
            return [i+1, result]

        batch_size = tf.shape(a_perm)[1]
        shape_invariants = [idx.get_shape(), tf.TensorShape([None, None, self.num_perspectives])]
        result = tf.zeros(shape=[1, batch_size, self.num_perspectives])
        result = tf.while_loop(cond, body, [idx, result], shape_invariants=shape_invariants) # A_time_steps x batch x perspectives
        return tf.transpose(result[1][1:, :, :], perm=[1, 0, 2]) # batch x time_steps x perspective

def numpy_reference_fma(A, B, W, batch_size, A_time_steps, hidden_size, num_perspectives):
    result = np.zeros([A_time_steps, batch_size, num_perspectives])
    for i in range(0, A_time_steps):
        a_slice = np.expand_dims(A[:, 1, :], axis=1)
        b_slice = np.expand_dims(B[:, -1, :], axis=1)
        h_i = np.transpose(a_slice, (1, 0, 2))
        h_n = np.transpose(b_slice, (1, 0, 2))
        result[i] = numpy_reference_compute_score(h_i, h_n, W, batch_size, hidden_size, num_perspectives)
    return np.transpose(result, (1, 0, 2))


if __name__ == "__main__":
    with tf.Session() as session:
        batch_size = 3
        hidden_size = 4
        num_perspectives = 2
        A_time_steps = 5
        B_time_steps = 6

        A, B, W  = numpy_generate_A_B_w(batch_size, hidden_size, A_time_steps, B_time_steps, num_perspectives)
        fma = Full_Matching_Attention_Layer(num_perspectives)
        score_fn = fma.compute_attention(tf.constant(A, dtype=tf.float32), tf.constant(B, dtype=tf.float32), tf.constant(W, dtype=tf.float32))
        score = session.run(score_fn)        
        reference = numpy_reference_fma(A, B, W, batch_size, A_time_steps, hidden_size, num_perspectives)
        assert score.all() == reference.all()
