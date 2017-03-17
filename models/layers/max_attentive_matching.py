from attention_base_class import Attention_Base_Class, numpy_reference_compute_score, numpy_generate_A_B_w, numpy_check_equality
from attentive_matching_layer import numpy_compute_alpha_embeddings
import sys
sys.path.insert(0, '../')
import tensorflow as tf
from util import multiply_3d_by_2d, cosine_similarity
import numpy as np

class Max_Attentive_Matching_Layer(Attention_Base_Class):
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
        batch_size = h_i.get_shape().as_list()[1]
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
        shape_invariants = [idx.get_shape(), tf.TensorShape([batch_size, None,])]
        result = tf.while_loop(cond, body, [idx, result_init], shape_invariants=shape_invariants)
        result = result[1][:, 1:]
        return result

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
        hidden_size = a.get_shape()[2]
        b_perm = tf.transpose(b, perm=[1,2,0])
        b_time_steps = b_perm.get_shape()[0]

        idx = tf.constant(0) # Current time step index
        cond = lambda i, result: i < a_time_steps
        def body(i, result):
            h_i = a_perm[i, :, :] #hidden_size x batch
            alpha = self.compute_alpha_embeddings(h_i, b_perm) # batch x B_time_steps
            max_indices = tf.argmax(alpha, axis=1) # batch x 1 (time_step index)
            dense_max = tf.expand_dims(tf.one_hot(max_indices, b_time_steps), axis=1) # batch x 1 x b_time_steps
            h_i_mean = tf.transpose(tf.matmul(dense_max, b), perm=[1, 0, 2]) # 1 x batch x hidden
            h_i_perm = tf.expand_dims(tf.transpose(h_i), 0) # 1 x batch size x hidden
            m_i = tf.expand_dims(self.compute_score(h_i_perm, h_i_mean, W), axis=0) # 1 x batch x perspectives
            result = tf.concat([result, m_i], axis=0) # building time_steps x batch x persspective
            return [i+1, result]

        batch_size = tf.shape(a_perm)[2]
        shape_invariants = [idx.get_shape(), tf.TensorShape([None, None, self.num_perspectives])]
        result = tf.zeros(shape=[1, batch_size, self.num_perspectives])
        result = tf.while_loop(cond, body, [idx, result], shape_invariants=shape_invariants) # A_time_steps x batch x perspectives
        return tf.transpose(result[1][1:, :, :], perm=[1, 0, 2]) # batch x A_time_steps x perspective

def numpy_reference_max_aml(A, B, W, A_time_steps, B_time_steps, batch_size, num_perspectives, hidden_size):
    A_perm = np.transpose(A, [1,2,0])
    B_perm = np.transpose(B, [1,2,0])
    result = np.zeros([A_time_steps, batch_size, num_perspectives])
    for i in range(0, A_time_steps):
        h_i = A_perm[i, :, :]
        alpha = numpy_compute_alpha_embeddings(h_i, B_perm, B_time_steps, batch_size, hidden_size) #batch x b_time_steps
        result_i = np.zeros([batch_size, hidden_size])
        for j in range(0, batch_size):
            best_time_step = np.argmax(alpha[j])
            result_i[j] = B[j][best_time_step]
        h_i_perm = np.expand_dims(np.transpose(h_i), 0)
        m_i = numpy_reference_compute_score(h_i_perm, result_i, W, batch_size, hidden_size, num_perspectives) # 1 x batch x perspectives
        result[i, :, :] = m_i
    result = np.transpose(result, (1, 0, 2))
    return result


if __name__ == "__main__":
    batch_size = 3
    hidden_size = 4
    num_perspectives = 2
    A_time_steps = 5
    B_time_steps = 6

    with tf.Session() as session:
        batch_size = 3
        hidden_size = 4
        num_perspectives = 2
        A_time_steps = 5
        B_time_steps = 6

        A, B, W  = numpy_generate_A_B_w(batch_size, hidden_size, A_time_steps, B_time_steps, num_perspectives)

        aml = Max_Attentive_Matching_Layer(num_perspectives)
        score_fn = aml.compute_attention(tf.constant(A, dtype=tf.float32), tf.constant(B, dtype=tf.float32), tf.constant(W, dtype=tf.float32))
        score = session.run(score_fn)
        reference = numpy_reference_max_aml(A, B, W, A_time_steps, B_time_steps, batch_size, num_perspectives, hidden_size)
        assert numpy_check_equality(score, reference, 1e-4)
