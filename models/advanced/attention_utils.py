#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
CS 224N 2016-2017
attention_utils.py: Utilty classes for Conditional LSTM With Attention
Sahil Chopra <schopra8@cs.stanford.edu>
Saachi Jain <saachi@cs.stanford.edu>
John Sholar <jmsholar@cs.stanford.edu>
"""

import tensorflow as tf
import numpy as np
import collections
from tensorflow.python.ops.math_ops import tanh
from tensorflow.contrib.rnn import LSTMCell


# TODO: Implement BlockAttentionLSTMCell with monolithic ops for efficiency
# See https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/LSTMBlockCell

class AttentionLSTMCell(LSTMCell):
    """
    Implementation of LSTM with Attention, as described in Rocktaschel et. al.,
    2016 (https://arxiv.org/pdf/1509.06664.pdf)
    Extends Tensorflow's LSTMCell Class tensorflow.contrib.rnn.LSTMCell
        Source Code:
            https://github.com/tensorflow/tensorflow/blob/master/tensorflow/
            contrib/rnn/python/ops/core_rnn_cell_impl.py#L252
        Documentation:
            https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/LSTMCell
    """

    # Follows initialization of superclass, with additional required parameters:
    #   sequence_length: length of sequence RNN decodes
    def __init__(self, num_units, sequence_length, input_size=None,
                 use_peepholes=False, cell_clip=None,
                 initializer=None, num_proj=None, proj_clip=None,
                 num_unit_shards=None, num_proj_shards=None,
                 forget_bias=1.0, state_is_tuple=True,
                 activation=tanh, reuse=None):
        if not state_is_tuple:
            raise Exception('state_is_tuple = False not supported')
        self._sequence_length = sequence_length
        self._state_size = AttentionLSTMStateTuple(
            num_units, num_units, (sequence_length, num_units),
        )

    def __call__(self, inputs, state, scope=None):
        # Extract state tuple from previous iteration
        #   c_prev: cell state vector for the previous iteration
        #   m_prev: exposed state vector for the previous iteration
        #   m_all: [sequence_length, num_units] matrix of all state vectors
        (c_prev, h_prev, h_all, index) = state
        state_reduced = (c_prev, h_prev)
        new_m, new_state_reduced = super(AttentionLSTMCell, self).__call__(
            self, inputs, state_reduced)
        new_c, new_h = new_state_reduced
        h_all[index, :] = new_h
        index += 1
        new_state = new_c, new_h, h_all, index
        return new_m, new_state

    def zero_state(self, batch_size, dtype):
        c, h = super(AttentionLSTMCell, self).zero_state(
            self, batch_size, dtype)
        # is initializing using np.zeros appropriate?
        m = np.zeros(self._sequence_length, self._num_units)
        index = 0
        return c, h, m, index


_AttentionLSTMStateTuple = collections.namedtuple("AttentionLSTMStateTuple",
                                                  ("c", "h", "h_all", "index"))


class AttentionLSTMStateTuple(_AttentionLSTMStateTuple):
    """Tuple used by Attention LSTM Cells for
    `state_size`, `zero_state`, and output state.
    Stores four elements: `(c, h, h_all, index)`, in that order.
        c: current cell state vector
        h: current exposed state vector
        h_all: matrix of all exposed state vectors from all previous states
        index: temporal index of current state (begins at 0)
    Only used when `state_is_tuple=True`.
    """
    __slots__ = ()

    @property
    def dtype(self):
        (c, h) = self
        if not c.dtype == h.dtype:
          raise TypeError("Inconsistent internal state: %s vs %s" %
                          (str(c.dtype), str(h.dtype)))
        return c.dtype