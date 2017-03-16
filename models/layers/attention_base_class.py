import sys
sys.path.insert(0, '../')
import tensorflow as tf
from util import multiply_3d_by_2d, cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity

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
        batch_size = tf.shape(v1)[1]
        # assert hidden_size == v2.get_shape()[2]
        print v1.get_shape()
        print v2.get_shape()
        # assert batch_size == tf.shape(v2)[1]
        v1_collapse = tf.reshape(v1, shape=[hidden_size, batch_size]) # hidden x batch
        v2_collapse = tf.reshape(v2, shape=[hidden_size, batch_size]) # hidden x batch
        result = []
        idx = tf.constant(0)

        cond = lambda i: tf.less(i, self.num_perspectives)
        def body(i):
            print "run"
            W_i = tf.expand_dims(W[:, i], axis=1) # hidden x 1
            v_1_w_i = tf.multiply(v1_collapse, W_i) # hidden x batch
            v_2_w_i = tf.multiply(v2_collapse, W_i) # hidden x batch
            result.append(cosine_similarity(v_1_w_i, v_2_w_i)) # batch x 1
            return tf.add(i, 1)
        #shape invariants
        shape_invariants = [idx.get_shape()]
        print "about to run"
        loop_perspectives = tf.while_loop(cond, body, [idx], shape_invariants = shape_invariants)
        print "done_running"
        result = tf.stack(result, axis=1)
        print "done stacking"
        return result

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

if __name__ == "__main__":
    with tf.Session() as session:
        # batch_size: 3
        # hidden_size: 4
        # perspectives: 2
        v1 = tf.constant([[
            [1,2,3,4],
            [5,6,7,8],
            [9,10,11,12]]], dtype=tf.float32) # 1 x batch x hidden
        v2 = tf.constant([[
            [1,2,10,4],
            [5,6,7,1],
            [9,13,11,12]]], dtype=tf.float32) # 1 x batch x hidden
        W = tf.constant([
            [1,2],
            [3,4],
            [5,6],
            [7,8]], dtype=tf.float32)
        abc = Attention_Base_Class(2)
        score = session.run(abc.compute_score(v1, v2, W))
        print "out of session"
        # checking our work:
        v1_w1 = np.array([
            [1, 5, 9],
            [6, 18, 30],
            [15, 35, 55],
            [28, 56, 84]])

        v1_w2 = np.array([
            [2, 10, 18],
            [8, 24, 40],
            [18, 42, 66],
            [32, 64, 96]])

        v2_w1 = np.array([
            [1, 5, 9],
            [6, 18, 39],
            [50, 35, 55],
            [28, 7, 84]])

        v2_w2 = np.array([
            [2, 10, 18],
            [8, 24, 52],
            [66, 42, 66],
            [32, 8, 96]])

        print "computing similarities"
        cos_sim_w1 = sk_cosine_similarity(v1_w1, v2_w1)
    # cos_sim_w2 = sk_cosine_similarity(v1_w2, v2_w2)
    # full = np.concat(cos_sim_w1, cos_sim_w2, axis=1)
    # print score
    # print full
    # assert score == full
