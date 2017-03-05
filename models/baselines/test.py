#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
CS 224N 2016-2017 
util.py: General utility routines.
Sahil Chopra <schopra8@cs.stanford.edu>
Saachi Jain <saachi@cs.stanford.edu>
John Sholar <jmsholar@cs.stanford.edu>
"""
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import time
import itertools
import random
import os
import sys
sys.path.insert(0, '../')

from fnc1_utils.score import report_score
from model import Model
from fnc1_utils.featurizer import read_binaries
from util import vectorize_stances

class Config:
    """Holds model hyperparams and data information.
    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation. Use self.config.? instead of Config.?
    """
    # TODO: Change input_dim and num_classes appropriately (currently set for training example)
    initial_embedding_dim = 2
    reduced_embedding_dim = 1
    input_dim = 1

    num_classes = 2
    dropout = 0.5
    batch_size = 1000
    lr = .01
    n_epochs = 100

class Test_Neural_Network(Model):
    """ Simple Test Neural Network (based on SNLI Corpus NN)
    """

    def __init__(self, config):
        self.config = config

        # Defining placeholders.
        self.inputs_placeholder = None
        self.labels_placeholder = None
        self.dropout_placeholder = None

        # Matrix Names
        self.weight_names = ['W1', 'W2', 'W3']
        self.bias_names = ['b1', 'b2', 'b3']

        self.build()

    def add_placeholders(self):
        """ Generates placeholder variables to represent the input tensors.
        """
        self.inputs_placeholder = tf.placeholder(tf.float32, (None, self.config.initial_embedding_dim))
        self.headlines_placeholder = tf.placeholder(tf.float32, (None, self.config.initial_embedding_dim))
        self.labels_placeholder = tf.placeholder(tf.float32, (None, self.config.num_classes))
        self.dropout_placeholder = tf.placeholder(tf.float32)

    def create_feed_dict(self, inputs_batch, labels_batch=None, dropout=1):
        """Creates the feed_dict for the model.
        """
        if labels_batch is not None:
            feed_dict = {
                self.inputs_placeholder: inputs_batch,
                self.dropout_placeholder: dropout,
                self.labels_placeholder: labels_batch,
            }

        else:
             feed_dict = {
                self.inputs_placeholder: inputs_batch,
                self.dropout_placeholder: dropout,
              }

        return feed_dict

    def add_prediction_op(self): 
        """Runs the SNLI Basline NN on the input.

        Returns:
            preds: tf.Tensor of shape (batch_size, 1)
        """

        # Declare initial tf.Variable matrices to map from initial embeddings to reduced dimension embeddings
        init_article_weights = tf.get_variable("init_article_weights", 
            shape=[self.config.initial_embedding_dim, self.config.reduced_embedding_dim],
            initializer=tf.contrib.layers.xavier_initializer()
        )

        init_article_bias = tf.get_variable("init_article_bias",
            shape=[self.config.reduced_embedding_dim],
            initializer=tf.constant_initializer(0)
        )

        # Construct joint article + headline embedding
        embedding = tf.nn.tanh(tf.matmul(self.inputs_placeholder, init_article_weights) + init_article_bias)

        # Declare tf.Variable weight matrices and bias vectors for 3 tanh layers
        weights = [
            tf.get_variable(n,
                shape=[self.config.input_dim, self.config.input_dim],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            for n in self.weight_names
        ]

        biases = [
            tf.get_variable(n,
                shape=[self.config.input_dim],
                initializer=tf.constant_initializer(0),
            )
            for n in self.bias_names
        ]

        # Declare tf.Variable weight matrix and bias vector for final layer
        final_weights = tf.get_variable("final_weights",
            shape=[self.config.input_dim, self.config.num_classes],
            initializer=tf.contrib.layers.xavier_initializer()
        )

        final_biases = tf.get_variable("final_biases",
            shape=[self.config.num_classes],
            initializer=tf.constant_initializer(0),
        )

        # Declare intermediate variables as products of base variables for tanh layers and final layer
        layer_1 = tf.nn.tanh(tf.matmul(embedding, weights[0]) + biases[0])
        layer_2 = tf.nn.tanh(tf.matmul(layer_1, weights[1]) + biases[1])
        layer_3 = tf.nn.tanh(tf.matmul(layer_2, weights[2]) + biases[2])
        final_layer = tf.matmul(layer_3, final_weights) + final_biases

        return final_layer

    def add_loss_op(self, final_layer):
        """Adds Ops for the loss function to the computational graph.

        Args:
            pred: A tensor of shape (batch_size, n_classes) containing the output of the neural
                  network before the softmax layer.
        Returns:
            loss: A 0-d tensor (scalar)
        """
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=self.labels_placeholder,
                logits=final_layer)
        )
        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train.

        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        optimizer = tf.train.AdamOptimizer(self.config.lr)
        train_op = optimizer.minimize(loss)
        return train_op    

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        feed = self.create_feed_dict(
            inputs_batch,
            labels_batch=labels_batch,
            dropout=self.config.dropout
        )

        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        """Perform one step of gradient descent on the provided batch of data.

        Args:
            sess: tf.Session()
            inputs_batch: np.ndarray of shape (n_samples, n_dim)
            labels_batch: np.ndarray of shape (n_samples, n_classes)
        Returns:
            loss: loss over the batch (a scalar)
        """
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def predict_on_batch(self, sess, inputs_batch):
        """Make predictions for the provided batch of data

        Args:
            sess: tf.Session()
            inputs_batch: np.ndarray of shape (n_samples, n_dim)
        Returns:
            predictions: np.ndarray of shape (n_samples, n_classes)
        """
        feed = self.create_feed_dict(inputs_batch)
        predictions = sess.run(self.pred, feed_dict=feed)
        preds = np.argmax(predictions, axis=1)

        return preds

    def run_epoch(self, sess, train_examples, dev_set):
        prog = Progbar(target=1 + train_examples[0].shape[0] / self.config.batch_size)
        for i, (inputs_batch, labels_batch) in enumerate(minibatches(train_examples, self.config.batch_size)):
            loss = self.train_on_batch(sess, inputs_batch, labels_batch)
            prog.update(i + 1, [("train loss", loss)])

    def fit(self, sess, saver, train_examples):
        best_dev_score = 0
        for epoch in range(self.config.n_epochs):
            print "Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs)

def main(debug=True):
    # Generate Sample Dataset in 2-dimensional space
    # Class 1 is a Gaussian Centered Around (0, -5)
    # Class 2 is a Gaussian Centered Around (0, 5)

    centroid_1 = np.array([0, -5])
    centroid_2 = np.array([0, 5])
    cov = np.array([
      [1, 0],
      [0, 1]
    ])
    size = 500
    x1, y1 = np.random.multivariate_normal(centroid_1, cov, size).T
    x2, y2 = np.random.multivariate_normal(centroid_2, cov, size).T
    labels_1 = np.concatenate([np.array([[1, 0]]) for _ in range(size)], axis=0)
    labels_2 = np.concatenate([np.array([[0, 1]]) for _ in range(size)], axis=0)
    x = np.concatenate([x1, x2], axis = 0).reshape((-1, 1))
    y = np.concatenate([y1, y2], axis = 0).reshape((-1, 1))

    # Plot data
    plt.plot(x1, y1, 'x')
    plt.plot(x2, y2, 'x')
    plt.axis('equal')
    plt.savefig('plot.png', bbox_inches='tight')

    all_data = np.concatenate([x, y], axis=1)
    all_labels = np.concatenate([labels_1, labels_2], axis=0)

    # Split example data into train and test sets
    train_data, test_data, train_labels, test_labels = train_test_split(
      all_data, all_labels, test_size=0.2, random_state=42
    )  

    with tf.Graph().as_default():
        print 80 * "="
        print "INITIALIZING"
        print 80 * "="
        config = Config()
        model = Test_Neural_Network(config)

        print "Building model...",
        start = time.time()
        print "took {:.2f} seconds\n".format(time.time() - start)

        init = tf.global_variables_initializer()
        saver = None if debug else tf.train.Saver()

        with tf.Session() as session:
            session.run(init)

            print 80 * "="
            print "TRAINING"
            print 80 * "="
            model.fit(session, saver, [train_data, train_labels])
            preds = model.predict_on_batch(session, test_data)

            print "-----"
            print preds
            print "-----"
            stances = np.array(vectorize_stances(test_labels))
            print stances

            print "-----"
            print "dif"
            print sum(preds - stances)


if __name__ == '__main__':
    main(False)
