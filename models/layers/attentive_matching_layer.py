import numpy as np
from attention_base_class import Attention_Base_Class, numpy_reference_compute_score, numpy_generate_A_B_w, numpy_check_equality
from scipy.spatial.distance import cosine as sk_cosine

import sys
sys.path.insert(0, '../')
import tensorflow as tf
from util import multiply_3d_by_2d, cosine_similarity

class Attentive_Matching_Layer(Attention_Base_Class):
    """
    """
    def compute_alpha_embeddings(self, h_i, H_q):
        """
        Args:
            h_i: tensor [hidden_size, batch_size]
            H_q: tensor [B_time_steps, hidden_size, batch]
        Output:
            alpha: tensor[batch_size, B_time_steps]
            We compute alpha_(i,j) = cosine(h_i, h_j_q) between
            a slice at a single timestep i in H_p (h_i) and all the timesteps
            in H_q.
        """
        batch_size = tf.shape(h_i)[1]
        hidden_size = h_i.get_shape().as_list()[0]
        b_time_steps = H_q.get_shape().as_list()[0]

        def body(j, result):
            h_j = H_q[j, :, :] # hidden_size x batch
            alpha = cosine_similarity(h_i, h_j)
            result = tf.concat([result, alpha], axis=1)
            return [j+1, result]

        idx = tf.constant(0) # Current time step index
        cond = lambda j, result: j < b_time_steps  
        result_init = tf.zeros([batch_size, 1]) # batch_size x 1
        shape_invariants = [idx.get_shape(), tf.TensorShape([None, None])]
        result = tf.while_loop(cond, body, [idx, result_init], shape_invariants=shape_invariants)
        result = result[1][:, 1:]
        return result

    def compute_h_i_mean(self, alpha, H_q):
        """
        Args:
            alpha: tensor [batch_size, B_time_steps]
            H_q: tensor [batch, B_time_steps, hidden_size]
        Output:
            h_i_mean: tensor[1, batch_size, hidden size]
        """
        alpha_sum = tf.reduce_sum(alpha, axis=1) # batch_size x 1
        b_time_steps = H_q.get_shape().as_list()[1]
        hidden_size = H_q.get_shape().as_list()[1]
        batch_size = tf.shape(H_q)[2]

        H_q_t = tf.transpose(H_q, perm=[0, 2, 1]) # batch x hidden_size x B_time_steps
        alpha_expand = tf.expand_dims(alpha, axis=2) # batch x B_time_steps x 1
        result = tf.matmul(H_q_t, alpha_expand) # batch x hidden x 1
        return tf.transpose(result, perm=[2, 0, 1]) # 1 x batch x hidden

    def compute_attention(self, a, b, W):
        """
        Args:
            a: tensor [batch, A_time_steps, hidden_size]
            b: tensor [batch, B_time_steps, hidden_size]
            W: weight matrix [hidden_size x num_perspectives]
        Returns:
            output: tensor [batch x A_time_steps x num_perspectives]
        """
        # a_perm is now [A_time_steps, hidden_size, batch]
        a_perm = tf.transpose(a, perm=[1, 2, 0])
        a_time_steps = a_perm.get_shape()[0]

        # b_perm is now [B_time_steps, hidden_size, batch]
        b_perm = tf.transpose(b, perm=[1,2,0])

        idx = tf.constant(0) # Current time step index
        cond = lambda i, result: i < a_time_steps
        def body(i, result):
            h_i = a_perm[i, :, :] #hidden_size x batch
            alpha = self.compute_alpha_embeddings(h_i, b_perm) # batch x B_time_steps
            h_i_mean = self.compute_h_i_mean(alpha, b) # 1 x batch_size x hidden
            h_i_perm = tf.expand_dims(tf.transpose(h_i), 0) # 1 x batch size x hidden
            m_i = tf.expand_dims(self.compute_score(h_i_perm, h_i_mean, W), axis=0) # 1 x batch x perspectives
            result = tf.concat([result, m_i], axis=0) # building time_steps x batch x persspective
            return [i+1, result]

        batch_size = tf.shape(a_perm)[2]
        shape_invariants = [idx.get_shape(), tf.TensorShape([None, None, self.num_perspectives])]
        result = tf.zeros(shape=[1, batch_size, self.num_perspectives])
        result = tf.while_loop(cond, body, [idx, result], shape_invariants=shape_invariants) # A_time_steps x batch x perspectives
        return tf.transpose(result[1][1:, :, :], perm=[1, 0, 2]) # batch x A_time_steps x perspective

# TESTING

def numpy_reference_aml(A, B, W, A_time_steps, B_time_steps, batch_size, num_perspectives, hidden_size):
    A_perm = np.transpose(A, [1,2,0])
    B_perm = np.transpose(B, [1,2,0])
    result = np.zeros([A_time_steps, batch_size, num_perspectives])

    for i in range(0, A_time_steps):
        h_i = A_perm[i, :, :]
        alpha = numpy_compute_alpha_embeddings(h_i, B_perm, B_time_steps, batch_size, hidden_size)
        h_i_mean = numpy_compute_h_i_mean(alpha, B, B_time_steps, hidden_size, batch_size)
        h_i_perm = np.expand_dims(np.transpose(h_i), 0)
        m_i = numpy_reference_compute_score(h_i_perm, h_i_mean, W, batch_size, hidden_size, num_perspectives) # 1 x batch x perspectives
        result[i, :, :] = m_i
    return result

def numpy_compute_alpha_embeddings(h_i, H_q, b_time_steps, batch_size, hidden_size):
    h_i_collapse = np.reshape(h_i, [hidden_size, batch_size])    
    result = np.zeros([batch_size, b_time_steps])

    for j in range(0, b_time_steps):
        h_j = H_q[j, :, :] # 1 x hidden_size x batch
        h_j_collapse = np.reshape(h_j, [hidden_size, batch_size])

        for k in range(0, batch_size):
            result[k, j] = 1-sk_cosine(h_i_collapse[:, k], h_j_collapse[:, k])

    return result

def numpy_compute_h_i_mean(alpha, H_q, b_time_steps, hidden_size, batch_size):
    alpha_sum = np.sum(alpha, axis=1) # batch_size x 1
    H_q_t = np.transpose(H_q, [0, 2, 1]) # batch x hidden_size x B_time_steps
    alpha_expand = np.expand_dims(alpha, axis=2) # batch x B_time_steps x 1
    result = np.matmul(H_q_t, alpha_expand) # batch x hidden x 1
    return np.transpose(result, [2, 0, 1]) # 1 x batch x hidden

def test():
    with tf.Session() as session:
        batch_size = 3
        hidden_size = 4
        num_perspectives = 2
        A_time_steps = 5
        B_time_steps = 6

        A, B, W  = numpy_generate_A_B_w(batch_size, hidden_size, A_time_steps, B_time_steps, num_perspectives)

        aml = Attentive_Matching_Layer(num_perspectives)
        score_fn = aml.compute_attention(tf.constant(A, dtype=tf.float32), tf.constant(B, dtype=tf.float32), tf.constant(W, dtype=tf.float32))
        score = session.run(score_fn)   
        reference = numpy_reference_aml(A, B, W, A_time_steps, B_time_steps, batch_size, num_perspectives, hidden_size)
        assert numpy_check_equality(score, reference, 1e-7)

if __name__ == '__main__':
    test()
