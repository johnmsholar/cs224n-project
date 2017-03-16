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

    def compute_score(self, v1, v2, W):
        """
        Args:
            v1: 1 x batch_size x hidden_size
            v2: 1 x batch_size x hidden_size
            W: Scoring Weight Matrix [hidden_size x num_perspectives]
        Returns:
            score: batch x perspectives
        """
        hidden_size = v1.get_shape().as_list()[2]
        batch_size = v1.get_shape().as_list()[1]
        # batch_size = tf.shape(v1)[1]
        v1_collapse = tf.transpose(tf.reshape(v1, shape=[batch_size, hidden_size])) # hidden x batch
        v2_collapse = tf.transpose(tf.reshape(v2, shape=[batch_size, hidden_size])) # hidden x batch
        result_init = tf.zeros(shape=[batch_size, 1])
        idx = tf.constant(0)

        cond = lambda i, result: tf.less(i, self.num_perspectives)
        def body(i, result):
            W_i = tf.expand_dims(W[:, i], axis=1) # hidden x 1
            v_1_w_i = tf.multiply(v1_collapse, W_i) # hidden x batch
            v_2_w_i = tf.multiply(v2_collapse, W_i) # hidden x batch
            cos_sim = cosine_similarity(v_1_w_i, v_2_w_i) # batch x 1
            result = tf.concat([result, cos_sim], axis=1)
            return [tf.add(i, 1), result]
        #shape invariants
        shape_invariants = [idx.get_shape(), tf.TensorShape([None, None])]
        print "about to run"
        loop_perspectives = tf.while_loop(cond, body, [idx, result_init], shape_invariants = shape_invariants)
        return loop_perspectives[1][:, 1:]

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

def numpy_reference_compute_score(v1, v2, W, batch_size, hidden_size, num_perspectives):
    result = np.zeros([batch_size,num_perspectives])
    for i in range(0, num_perspectives):
        w_sing = W[:, i] # hidden x 1
        v1_wi = np.transpose(v1*w_sing)
        v2_wi = np.transpose(v2*w_sing)
        for j in range(0, batch_size):
            result[j, i] = 1-sk_cosine(v1_wi[:, j], v2_wi[:, j])
    return result        

if __name__ == "__main__":
    with tf.Session() as session:
        batch_size = 3
        hidden_size = 4
        num_perspectives = 2

        v1 = [[1,2,3,4],
            [5,6,7,8],
            [9,10,11,12]]
        v2 = [[1,2,10,4],
            [5,6,7,1],
            [9,13,11,12]]
        W = [[1,2],
            [3,4],
            [5,6],
            [7,8]]

        v1_tf = tf.constant([v1], dtype=tf.float32) # 1 x batch x hidden
        v2_tf = tf.constant([v2], dtype=tf.float32) # 1 x batch x hidden
        W_tf = tf.constant(W, dtype=tf.float32)
        abc = Attention_Base_Class(num_perspectives)
        score_fn = abc.compute_score(v1_tf, v2_tf, W_tf)
        score = session.run(score_fn)
        # checking our work:
        v1_np = np.array(v1) # 1 x batch x hidden
        v2_np = np.array(v2) # 1 x batch x hidden
        W_np = np.array(W)
        ref_score = numpy_reference_compute_score(v1_np, v2_np, W_np, batch_size, hidden_size, num_perspectives)
        assert ref_score.all() == score.all()