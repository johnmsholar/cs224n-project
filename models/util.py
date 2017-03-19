#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
CS 224N 2016-2017 
util.py: General utility routines.
Sahil Chopra <schopra8@cs.stanford.edu>
Saachi Jain <saachi@cs.stanford.edu>
John Sholar <jmsholar@cs.stanford.edu>
"""

import argparse
import sys
import time
import numpy as np
import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import tensorflow as tf

def vectorize_stances(stances):
    v_stances = []
    for row in stances:
        s = np.where(row == 1.0)
        v_stances.append(s[0].tolist()[0])
    return v_stances

def get_minibatches(data, minibatch_size, shuffle=True):
    """
    Iterates through the provided data one minibatch at at time. You can use this function to
    iterate through data in minibatches as follows:

        for inputs_minibatch in get_minibatches(inputs, minibatch_size):
            ...

    Or with multiple data sources:

        for inputs_minibatch, labels_minibatch in get_minibatches([inputs, labels], minibatch_size):
            ...

    Args:
        data: there are two possible values:
            - a list or numpy array
            - a list where each element is either a list or numpy array
        minibatch_size: the maximum number of items in a minibatch
        shuffle: whether to randomize the order of returned data
    Returns:
        minibatches: the return value depends on data:
            - If data is a list/array it yields the next minibatch of data.
            - If data a list of lists/arrays it returns the next minibatch of each element in the
              list. This can be used to iterate through multiple data sources
              (e.g., features and labels) at the same time.

    """
    list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
    data_size = len(data[0]) if list_data else len(data)
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    for minibatch_start in np.arange(0, data_size, minibatch_size):
        minibatch_indices = indices[minibatch_start:minibatch_start + minibatch_size]
        yield [minibatch(d, minibatch_indices) for d in data] if list_data \
            else minibatch(data, minibatch_indices)


def minibatch(data, minibatch_idx):
    return data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]

def minibatches(data, batch_size, shuffle=True):
    return get_minibatches(data, batch_size, shuffle)

class Progbar(object):
    """
    Progbar class copied from keras (https://github.com/fchollet/keras/)
    Displays a progress bar.
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=[], exact=[]):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]
        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current)/self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('='*(prog_width-1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.'*(self.width-prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if type(self.sum_values[k]) is list:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=[]):
        self.update(self.seen_so_far+n, values)


# Scikit-Learn Library Function for Converting a Confusion Matrix into a Plot
# Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.figure()
    confusion_matrix_backend(cm, classes,
                             normalize=normalize,
                             title=title,
                             cmap=cmap)
    plt.show()


def save_confusion_matrix(cm, classes, filename,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.figure()
    confusion_matrix_backend(cm, classes,
                             normalize=normalize,
                             title=title,
                             cmap=cmap)
    plt.savefig(filename, bbox_inches='tight')


def confusion_matrix_backend(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{0:.4f}'.format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def create_tensorflow_saver(exclude_names):
    train_vars = [var for var in tf.global_variables() if var.name not in exclude_names]
    # print "SAVER VARIABLES"
    # print [var.name for var in train_vars]
    return tf.train.Saver(train_vars)

# broadcasting util function
# three_tensor is dimensions: x_dim, y_dim, h_dim
# two_tensor is dimensions: h_dim, z_dim
# result is: x_dim, y_dim, z_dim
# h_dim and z_dim should be static
def multiply_3d_by_2d(three_tensor, two_tensor, y_dim_flat=True):
    # print tf.shape(three_tensor)
    three_tensor_shape = tf.shape(three_tensor)
    x_dim = tf.shape(three_tensor)[0]
    if y_dim_flat:
        y_dim = three_tensor.get_shape().as_list()[1]
    else:
        y_dim = three_tensor_shape[1]
    [h_dim, z_dim] = two_tensor.get_shape().as_list()

    # three_tensor is now x_dim*y_dim by h_dim
    reshaped_three_tensor = tf.reshape(three_tensor, [x_dim*y_dim, h_dim])
    # result is x_dim*y_dim by z_dim
    multiplied_tensor = tf.matmul(reshaped_three_tensor, two_tensor)
    packed_back_three_tensor = tf.reshape(multiplied_tensor, [x_dim, y_dim, z_dim])
    return packed_back_three_tensor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--restore', type=str, default=None)
    parser.add_argument('--test', type=bool, default=False)
    args = parser.parse_args()
    return args.epoch, args.restore

# given two matrices of same dimensions (a,b) compute cosine similarity
# broadcasted over y_dim
# adapted from tensorflow's word2vec implementation
# Args:
#   a, b: tensor of hidden x batch
# Returns: batch x 1 tensor that is cosine similarity between a and b over the batch sizes
def cosine_similarity(a, b):
    a_norm = tf.norm(a, axis=0) # dim: batch x 1
    b_norm = tf.norm(b, axis=0) # dim: batch x 1
    hidden_size = a.get_shape().as_list()[0]
    batch_size = tf.shape(a)[1]
    a_expand = tf.expand_dims(tf.transpose(a), axis=1) # batch, 1, hidden
    b_expand = tf.expand_dims(tf.transpose(b), axis=2) # bath, hidden, 1
    a_b_expand = tf.matmul(a_expand, b_expand) # batch x 1 x 1
    a_b = tf.reshape(a_b_expand, shape=[batch_size, 1]) #batch x 1
    a_b_norm = tf.norm(a_b, axis = 1) # batch x 1
    return tf.expand_dims(a_b_norm/(a_norm*b_norm), axis=1) # batch x 1

# # Given a matrix with [batch x time_steps x hidden] and a tensor of the 
# # For time_steps, batch with 0s in hidden in A, replace with given state
# # from batch
# # Args:
# #   A: [batch x time_steps x hidden]
# #   H: [batch x hidden] (the last hiddens for each batch example)
def extend_padded_matrix(A, H):
    hidden_size = A.get_shape().as_list()[2]
    norms = tf.norm(A, axis=2) # dim: batch x time_steps
    zeros = tf.to_float(tf.equal(norms, 0)) # dim: batch x time_steps, 1 where all 0s in hidden
    multiples = tf.constant([1, 1, hidden_size])
    exp_zeros = tf.tile(tf.expand_dims(zeros, axis=2), multiples) # dim: batch x time_steps x hidden_size
    zeros_transp = tf.transpose(exp_zeros, [1, 0, 2]) # time_steps x batch x hidden
    hidden_exp = tf.transpose(zeros_transp*H,  [1, 0, 2]) # batch x time_steps x hidden
    return A + hidden_exp

