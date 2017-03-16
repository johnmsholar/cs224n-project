from attention_base_class import Attention_Base_Class, cosine_similarity
import sys

sys.path.insert(0, '../')
import tensorflow as tf
from util import multiply_3d_by_2d

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
            We compute alpha_(i,j) = cosine(h_i, h_j_q) bewtween
            a slice at a single timestep i in H_p and all the timesteps
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
        result = [:, 1:]
        return result

    def compute_h_i_mean(alpha, H_q):
        """
        Args:
            alpha: tensor [batch_size, B_time_steps]
            H_q: tensor [B_time_steps, hidden_size, batch_size]
        Output:
            H_i_mean: tensor[B_time_steps, hidden_size, batch_size]
        """
        alpha_sum = tf.reduce_sum(alpha, axis=1) # batch_size x 1
        b_time_steps = H_q.get_shape().as_list()[0]
        hidden_size = H_q.get_shape().as_list()[1]
        batch_size = H_q.get_shape().as_list()[2]
        result_init = tf.zeros([1, hidden_size, batch_size])

        idx = tf.constant(0) # Current time step index
        cond = lambda j, result: j < b_time_steps
        def body(j, result):
            alpha_i_j = alpha[:, j] # 1 x batch_size
            h_j = H_q[j, :, :] # 1 x hidden_size x batch

            # TODO: Test Broadcasting operations in Tensorflow
            h_i_mean = tf.mul(alpha_i_j, h_j) / alpha_sum # 1 x hidden_size x batch
            
            result = tf.concat([result, h_i_mean], axis=0)
            return [j+1, result]
        result = tf.while_loop(cond, body, [idx, result_init])
        result = result[1,:,:]
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

        # b_perm is now [B_time_steps, hidden_size, batch]
        b_perm = tf.transpose(b, perm=[1, 2, 0])
        B_time_steps = b_perm.get_shape()[0]

        idx = tf.constant(0) # Current time step index
        cond = lambda i, result: i < a_time_steps
        def body(i, result):
            h_i = a[i, :, :] # 1 x hidden_size x batch
            alpha = compute_alpha_embeddings(h_i, b) # batch x B_time_steps
            h_i_mean = compute_h_i_mean(alpha, b) # B_time_steps x hidden_size x batch_size

            attention_computation_idx = tf.constant(0) # current time step index
            cond = lambda attention_computation_idx, result: attention_computation_idx < a_time_steps
            def body(attention_computation_idx, result):
                h_i = a_perm[i, :, :] # 1 x batch x hidden_size
                m_i = score(h_i, h_n, W) # 1 x batch x perspectives (score returns 3D)
                result = tf.concat([result, m_i], axis=0) # building batch x time_steps x persepctive
                return [i+1, result]

            batch_size = tf.shape(a_perm)[1]
            result = tf.zeros(shape=[1, batch_size, self.num_perspectives])
            tf.while_loop(cond, body, [idx, result])
            result = result[1:, :, :]    





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

