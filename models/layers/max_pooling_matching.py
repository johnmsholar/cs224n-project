from attention_base_class import Attention_Base_Class, numpy_reference_compute_score, numpy_check_equality, numpy_generate_A_B_w
import sys
import numpy as np

sys.path.insert(0, '../')
import tensorflow as tf
from util import multiply_3d_by_2d


class Max_Pooling_Attention_Layer(Attention_Base_Class):
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
        b_perm = tf.transpose(b, perm=[1, 0, 2])
        # h_n is [1 x batch x hidden_size]
        h_n =  tf.transpose(tf.expand_dims(b[:, -1, :], axis=0), perm=[1, 0, 2])

        idx = tf.constant(0) # current time step index
        cond = lambda i, result: tf.less(i, a_time_steps)
        def body(i, result):
            h_i = tf.expand_dims(a_perm[i, :, :], axis=0) # 1 x batch x hidden_size
            m_i = self.compute_max_elem_score(h_i, b_perm, W) # 1 x batch x perspectives
            result = tf.concat([result, m_i], axis=0) # building batch x time_steps x persepctive
            return [i+1, result]

        batch_size = tf.shape(a_perm)[1]
        shape_invariants = [idx.get_shape(), tf.TensorShape([None, None, self.num_perspectives])]
        result = tf.zeros(shape=[1, batch_size, self.num_perspectives])
        result = tf.while_loop(cond, body, [idx, result], shape_invariants=shape_invariants) # A_time_steps x batch x perspectives
        sliced_result = result[1][1:, :, :]
        return tf.transpose(result[1][1:, :, :], perm=[1, 0, 2]) # batch x time_steps x perspective

    def compute_max_elem_score(self, h_i, B, W, ):
        # Args:
        #   h_i is [1 x batch x hidden_size]
        #   B is [B_time_steps, batch x hidden_size]
        #   W is [hidden_size x num_perspectives]
        # Returns:
        #   1 x batch x perspectives which is max elem score over the scores on B
        b_time_steps = B.get_shape()[0]
        cond = lambda j, result: tf.less(j, b_time_steps)
        def body(j, result):
            h_j = tf.expand_dims(B[j, :, :], axis=0) # 1 x batch x hidden_size
            m_i_j = tf.expand_dims(self.compute_score(h_i, h_j, W), axis=0) # 1 x batch x perspectives
            result = tf.concat([result, m_i_j], axis=0)
            return [tf.add(j,1), result]
        batch_size = tf.shape(h_i)[1]
        jdx = tf.constant(0) # current j time step index
        shape_invariants = [jdx.get_shape(), tf.TensorShape([None, None, self.num_perspectives])]
        result = tf.zeros(shape=[1, batch_size, self.num_perspectives])
        result = tf.while_loop(cond, body, [jdx, result], shape_invariants=shape_invariants) # B_time_steps x batch x perspectives
        result = result[1][1:, :, :] # B_time_steps x batch x perspective
        result = tf.expand_dims(tf.reduce_max(result, axis=0), axis=0) # take elementwise max
        return result # 1 x batch x perspective

def numpy_reference_mpa(A, B, W, batch_size, A_time_steps, hidden_size, num_perspectives):
    # A [batch_size, A_time_steps, hidden_size]
    # B [batch_size, B_time_steps, hidden_size]
    # W [hidden_size, num_perspectives]
    result = np.zeros([A_time_steps, batch_size, num_perspectives])
    for i in range(0, A_time_steps):
        a_slice = np.expand_dims(A[:, i, :], axis=1)
        h_i = np.transpose(a_slice, (1, 0, 2))
        i_result = np.zeros([B_time_steps, batch_size, num_perspectives])
        for j in range(0, B_time_steps):
            b_slice = np.expand_dims(B[:, j, :], axis=1)
            h_j = np.transpose(b_slice, (1, 0, 2))
            i_result[j] = numpy_reference_compute_score(h_i, h_j, W, batch_size, hidden_size, num_perspectives)
        result[i] = np.amax(i_result, axis=0)
    return np.transpose(result, (1, 0, 2))
    return result

if __name__ == "__main__":
    with tf.Session() as session:
        batch_size = 3
        hidden_size = 4
        num_perspectives = 2
        A_time_steps = 5
        B_time_steps = 6

        A, B, W  = numpy_generate_A_B_w(batch_size, hidden_size, A_time_steps, B_time_steps, num_perspectives)
        mpa = Max_Pooling_Attention_Layer(num_perspectives)
        score_fn = mpa.compute_attention(tf.constant(A, dtype=tf.float32), tf.constant(B, dtype=tf.float32), tf.constant(W, dtype=tf.float32))
        score = session.run(score_fn)        
        reference = numpy_reference_mpa(A, B, W, batch_size, A_time_steps, hidden_size, num_perspectives)
        assert numpy_check_equality(score, reference, 1E-6)
