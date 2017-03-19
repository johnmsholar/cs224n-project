from attention_base_class import Attention_Base_Class, numpy_reference_compute_score, numpy_generate_A_B_w, numpy_check_equality
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

        h_n = tf.transpose(b[:, -1, :]) # hidden x batch
        a_M = tf.transpose(a, [1, 2, 0])
        result = self.compute_vec_matrix_score(h_n, a_M, W) # A_time_steps x p x b
        return tf.transpose(result, [2, 0, 1])

def numpy_reference_fma(A, B, W, batch_size, A_time_steps, hidden_size, num_perspectives):
    result = np.zeros([A_time_steps, batch_size, num_perspectives])
    h_n = np.transpose(B[:, -1, :])
    for i in range(0, A_time_steps):
        h_i = np.transpose(A[:, i, :])
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
        assert numpy_check_equality(score, reference, 1e-5)
