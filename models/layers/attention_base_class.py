import sys
import numpy as np
sys.path.insert(0, '../')
import tensorflow as tf
from util import multiply_3d_by_2d, cosine_similarity
from scipy.spatial.distance import cosine as sk_cosine

class Attention_Base_Class(object):
    """
    """
    def __init__(self, num_perspectives):
        self.num_perspectives = num_perspectives


    def compute_vec_matrix_score(self, v, M, W ):
        # Args:
        #   v is [hidden x batch]
        #   M is [M_time_steps, hidden_size x batch]
        #   W is [hidden_size x num_perspectives]
        # Returns:
        #   M_time_steps x p x b

        W = tf.expand_dims(tf.transpose(W), 2) # p x h x 1
        wv = W*v # p x h x b
        M_time_steps = M.get_shape().as_list()[0]
        # make copies along the B_timesteps dim
        W_exp = tf.tile(tf.expand_dims(W, 1), [1, M_time_steps, 1, 1]) # p x M_time_steps x h x 1
        wm = W_exp*M # p x M_time_steps x h x b

        wv_norm = tf.norm(wv, axis=1) # p x b
        wm_norm = tf.norm(wm, axis=2) # p x M_time_steps x b
        wm_norm_transp = tf.transpose(wm_norm, [1, 0, 2]) # M_time_steps x p x b
        norm_prod = wm_norm_transp*wv_norm # M_time_steps x p x b

        # make copies of wv
        wv_transp = tf.transpose(wv, [2, 0, 1]) # b x p x h
        wv_exp = tf.tile(tf.expand_dims(wv_transp, 0), [M_time_steps, 1, 1, 1]) # M_time_steps x b x p x h
        wv_exp = tf.expand_dims(wv_exp, 4) # M_time_steps x b x p x h x 1
        wm_exp = tf.expand_dims(tf.transpose(wm, [1, 3, 0, 2]), 3) # M_time_steps x b x p x 1 x h
        dot_prod = tf.squeeze(tf.matmul(wm_exp, wv_exp)) # M_time_steps x b x p
        dot_prod = tf.transpose(dot_prod, [0, 2, 1]) # M_time_steps x p x b

        full = tf.divide(dot_prod, norm_prod) # M_time_steps x p x b
        return full

    def compute_score(self, v1, v2, W):
        """
        Args:
            v1: hidden x batch
            v2: hidden x batch
            W: Scoring Weight Matrix [hidden_size x num_perspectives]
        Returns:
            score: batch x perspectives
        """
        W = tf.expand_dims(tf.transpose(W), 2) # p x h x 1
        Wv1 = W*v1 # p x h x b
        Wv2 = W*v2 # p x h x b
        norm_prod = tf.norm(Wv1, axis=1) * tf.norm(Wv2, axis=1) # dim: p x b

        # perform dot product
        Wv1_exp = tf.expand_dims(tf.transpose(Wv1, perm=[0, 2, 1]) , 2) # p x b x 1 x h
        Wv2_exp = tf.expand_dims(tf.transpose(Wv2, perm=[0, 2, 1]), 3) # p x b x h x 1
        dot_prod = tf.squeeze(tf.matmul(Wv1_exp, Wv2_exp)) # p x b
        
        return tf.transpose(tf.divide(dot_prod, norm_prod)) # b x p

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

            fw_attention = self.compute_attention(A_fw, B_fw, self.W1)
            bw_attention = self.compute_attention(A_bw, B_bw, self.W2)
            output = tf.concat([fw_attention, bw_attention], 2)

        return output

def numpy_reference_compute_score(v1, v2, W, batch_size, hidden_size, num_perspectives):
    #   v1: hidden_size, batch_size)
    #   v2: hidden_size, batch_size)
    #   W:  hidden_size, num_perspectives)
    result = np.zeros([batch_size,num_perspectives])
    for i in range(0, num_perspectives):
        w_sing = W[:, i] # 1 x hidden
        v1_wi = np.transpose(np.transpose(v1)*w_sing)
        v2_wi = np.transpose(np.transpose(v2)*w_sing)
        for j in range(0, batch_size):
            result[j, i] = 1-sk_cosine(v1_wi[:, j], v2_wi[:, j])
    return result        

def numpy_generate_A_B_w(batch_size, hidden_size, A_time_steps, B_time_steps, num_perspectives):
    A = np.zeros([batch_size, A_time_steps, hidden_size])
    B = np.zeros([batch_size, B_time_steps, hidden_size])
    W = np.zeros([hidden_size, num_perspectives])
    counter = 0
    for i in range(0, batch_size):
        for j in range(0, A_time_steps):
            for k in range(0, hidden_size):
                A[i,j,k] = counter
                counter += 1
    for i in range(0, batch_size):
        for j in range(0, B_time_steps):
            for k in range(0, hidden_size):
                B[i,j,k] = counter
                counter += 1
    for i in range(0, hidden_size):
        for j in range(0, num_perspectives):
            W[i,j] = counter
            counter += 1
    return A, B, W

def numpy_check_equality(score, actual, threshold):
    diff = np.absolute(score - actual)
    fulfill_thresh = np.less_equal(diff, threshold)
    return np.all(fulfill_thresh)

if __name__ == "__main__":
    with tf.Session() as session:
        batch_size = 3
        hidden_size = 4
        num_perspectives = 2

        v1 = np.random.rand(hidden_size, batch_size)
        v2 = np.random.rand(hidden_size, batch_size)
        W = np.random.rand(hidden_size, num_perspectives)

        v1_tf = tf.constant(v1, dtype=tf.float32) # 1 x batch x hidden
        v2_tf = tf.constant(v2, dtype=tf.float32) # 1 x batch x hidden
        W_tf = tf.constant(W, dtype=tf.float32)
        abc = Attention_Base_Class(num_perspectives)
        score_fn = abc.compute_score(v1_tf, v2_tf, W_tf)
        score = session.run(score_fn)
        # checking our work:
        ref_score = numpy_reference_compute_score(v1, v2, W, batch_size, hidden_size, num_perspectives)
        assert numpy_check_equality(score, ref_score, 1E-5)