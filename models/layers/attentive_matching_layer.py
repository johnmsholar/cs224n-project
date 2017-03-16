from attention_base_class import Attention_Base_Class, numpy_reference_compute_score, numpy_generate_A_B_w

import sys
sys.path.insert(0, '../')
import tensorflow as tf
from util import multiply_3d_by_2d, cosine_similarity

class Attentive_Matching_Layer(Attention_Base_Class):
    """
    """
    def compute_alpha_embeddings(h_i, H_q):
        """
        Args:
            h_i: tensor [1, hidden_size, batch_size]
            H_q: tensor [B_time_steps, hidden_size, batch_size]
        Output:
            alpha: tensor[batch_size, B_time_steps]
            We compute alpha_(i,j) = cosine(h_i, h_j_q) between
            a slice at a single timestep i in H_p (h_i) and all the timesteps
            in H_q.
        """
        batch_size = h_i.get_shape().as_list()[2]
        hidden_size = h_i.get_shape().as_list()[1]
        b_time_steps = H_q.get_shape().as_list()[0]
        result_init = tf.zeros([batch_size, 1]) # batch_size x 1

        # Cosine function requires dimensions hidden_size x batch
        h_i_collapse = tf.reshape(h_i, shape=[hidden_size, batch_size])    

        idx = tf.constant(0) # Current time step index
        cond = lambda j, result: j < b_time_steps
        def body(j, result):
            h_j = H_q[j, :, :] # 1 x hidden_size x batch
            h_j_collapse = tf.reshape(h_j, shape=[hidden_size, batch_size])
            alpha = cosine_similarity(h_i_collapse, h_j_collapse)
            result = tf.concat([result, alpha], axis=1)
            return [j+1, result]
        result = tf.while_loop(cond, body, [idx, result_init])
        result = result[1][:, 1:]
        return result

    def compute_h_i_mean(alpha, H_q):
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
        batch_size = H_q.get_shape().as_list()[2]

        H_q_t = tf.transpose(H_q, perms=[0, 2, 1]) # batch x hidden_size x B_time_steps
        alpha_expand = tf.expand_dims(alpha, axis=2) # batch x B_time_steps x 1
        result = tf.matmul(H_q_t, alpha_expand) # batch x hidden x 1
        return tf.transpose(result, perms=[2, 0, 1]) # 1 x batch x hidden

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

        idx = tf.constant(0) # Current time step index
        cond = lambda i, result: i < a_time_steps
        def body(i, result):
            h_i = a[i, :, :] # 1 x hidden_size x batch
            alpha = compute_alpha_embeddings(h_i, b) # batch x B_time_steps

            #1 x batch_size x hidden_size
            h_i_mean = compute_h_i_mean(alpha, b) # 1 x batch_size x hidden
            h_i_perm = tf.transpose(h_i, perms=[0, 2, 1])
            m_i = self.compute_score(h_i_perm, h_i_mean) # 1 x batch x perspectives



            m_i = tf.expand_dims(self.compute_attentio(h_i, h_i_mean, W, a_time_steps), axis=0)
            result = tf.concat([result, m_i], axis=0) # building time_steps x batch x persspective
            return [i+1, result]

        batch_size = tf.shape(a_perm)[2]
        shape_invariants = [idx.get_shape(), tf.TensorShape([None, None, self.num_perspectives])]
        result = tf.zeros(shape=[1, batch_size, self.num_perspectives])
        result = tf.while_loop(cond, body, [idx, result], shape_invariants=shape_invariants) # A_time_steps x batch x perspectives
        return tf.transpose(result[1][1:, :, :], perm=[1, 0, 2]) # batch x A_time_steps x perspective

def numpy_compute_alpha_embeddings(h_i, B, A_time_steps, B_time_steps, batch_size, num_perspectives):
    alpha_embeddings = np.zeros([batch_size, B_time_steps])
    for i in range(0, B_time_steps):
        h_i = A[i, :, :]


def test():
batch_size = 3
hidden_size = 4
num_perspectives = 2
A_time_steps = 5
B_time_steps = 6

A, B, W  = numpy_generate_A_B_w(batch_size, hidden_size, A_time_steps, B_time_steps, num_perspectives)
A_perm = np.transpose(A, [1,2,0])
B_perm = np.transpose(B, [1,2,0])






