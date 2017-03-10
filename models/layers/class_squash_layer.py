import sys
sys.path.insert(0, '../')
import tensorflow as tf
from util import multiply_3d_by_2d

class ClassSquashLayer:
    """Squash a hidden state into a class mapping
    """
    def __init__(self, hidden_size, num_classes):
        self.hidden_size = hidden_size
        self.num_classes = num_classes

    def __call__(self, hidden_output, scope=None):
        """
        Args:
            output: the output hidden_state that we want to squash [batch, hidden_size]
        Returns:
            preds: a prediction on the classes [batch, num_classes]
        """
        scope = scope or type(self).__name__

        with tf.variable_scope(scope):
            U = tf.get_variable("U", shape=[self.hidden_size, self.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("b", shape=[self.num_classes],
                initializer=tf.constant_initializer(0))
            
            preds = tf.matmul(hidden_output, U) + b
            new_size = preds.get_shape().as_list()

            assert new_size == [None, self.num_classes]
        return preds


