import sys
sys.path.insert(0, '../')
import tensorflow as tf
from util import multiply_3d_by_2d
class Multiperspective_Matching_A_to_B_Layer:
    """
    """
    def __init__(self, num_perspectives):
        self.num_perspectives = num_perspectives

    def __call__(self, A, B, scope=None):
        """
        Args:
            A: tuple of matrices(fw, bw) each one of which is [batch, A_time_steps, hidden_size]
            B: tuple of matrices(fw, bw) each one of which is [batch, B_time_steps, hidden_size]
            scope: is the name of the scope to be used when defining the variables inside.
        Returns:
            output: a tensor of size [batch x A_time_steps x (num_persepctives x 8)]
        """
        scope = scope or type(self).__name__

        with tf.variable_scope(scope, initializer=tf.contrib.layers.xavier_initializer()):
            # Output of each attention layer is [batch x A_time_steps x (num_perspectives x 2)]
            # Concatenation order should always be foward and then backward
            fma_layer = Full_Matching_Attention_Layer(self.num_persepctives)          
            fma_output = Full_Matching_Attention_Layer(A, B)

            max_pooling_layer = Max_Pooling_Attention_Layer(self.num_persepctives)
            max_pooling_output = max_pooling_layer(A, B)

            attentive_matching_layer = Atentive_Matching_Layer(self.num_persepctives)
            attentive_matching_output = attentive_matching_layer(A, B)

            max_attentive_matching_layer = Max_Attentive_Matching_Layer(self.num_persepctives)
            max_attentive_matching_output = max_attentive_matching_output(A, B)

            # Concatenate along 3rd axis (num_perspectives x 2) -> (num_perspectives x 8)
            final_output = tf.concat([fma_output, max_pooling_output, attentive_matching_output, max_attentive_matching_output], 2)

        return final_output

