#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
CS 224N 2016-2017 
conditional_lstm.py: Conditional LSTM Implementation
Sahil Chopra <schopra8@cs.stanford.edu>
Saachi Jain <saachi@cs.stanford.edu>
John Sholar <jmsholar@cs.stanford.edu>
"""

mport tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import cPickle
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
    num_classes = 4 # Number of classses for classification task.
    embed_size = 300 # Size of Glove Vectors

    # Hyper Parameters
    hidden_size = 300 # Hidden State Size
    batch_size = 100
    n_epochs = 40
    lr = 0.2
    max_grad_norm = 5.

class BasicLSTM(Model):
    """ Simple LSTM concatenating the article and headline.
        Encode the article and headline as one dense vector.
        Compute classification by taking softmax ove this dense vector.
    """

    def __init__(self, config):
        self.config = config

        # Defining placeholders.
        self.inputs_placeholder = None
        self.labels_placeholder = None
        self.dropout_placeholder = None

        self.build()

    def add_placeholders(self):
        """ Generates placeholder variables to represent the input tensors.
        """
        self.inputs_placeholder = tf.placeholder(tf.int32, (None, self.max_length))
        self.labels_placeholder = tf.placeholder(tf.float32, (None, self.config.num_classes))
        self.dropout_placeholder = tf.placeholder(tf.float32)

    def create_feed_dict(self, inputs_placeholder, labels_batch=None, dropout=1):
        """Creates the feed_dict for the model.
        """
        if labels_batch is not None:
            feed_dict = {
                self.inputs_placeholder: inputs_placeholder,
                self.dropout_placeholder: dropout,
                self.labels_placeholder: labels_batch,
            }

        else:
             feed_dict = {
                self.inputs_placeholder: inputs_placeholder,
                self.dropout_placeholder: dropout,
              }

        return feed_dict

    def add_embedding(self):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:

        Returns:
            embeddings: tf.Tensor of shape (None, max_length, embed_size)
        """
        pretrained_embeddings_tensor = tf.Variable(self.pretrained_embeddings)
        e = tf.nn.embedding_lookup_sparse(pretrained_embeddings_tensor, self.input_placeholder)
        embeddings = tf.reshape(e, shape=[-1, self.max_length, self.config.embed_size])          
        return embeddings

    def add_prediction_op(self): 
        """Runs RNN on the input. 

        Returns:
            preds: tf.Tensor of shape (batch_size, self.config.hidden_size)
        """
        # Lookup Glove Embeddings for the input words (e.g. one at each time step)
        x = self.add_embedding()
        dropout_rate = self.dropout_placeholder

        # Create final layer to project the output from th RNN onto
        # the four classification labels.
        U = tf.get_variable("U", shape=[self.config.hidden_size, self.config.num_classes],
            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("b", shape=[self.config.num_classes],
            initializer=tf.constant_initializer(0))

        # Compute the output at the end of the LSTM (automatically unrolled)
        cell = tf.nn.rnn_cell.LSTMCell(num_units=self.config.hidden_size)
        output, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32, sequence_length = self.sequence_lengths)

        # Compute predictions
        output_dropout = tf.nn.dropout(output, dropout_rate)
        preds = tf.matmul(U, output_dropout) + b
        return preds


    def add_loss_op(self, preds):
        """Adds Ops for the loss function to the computational graph.

        Args:
            pred: A tensor of shape (batch_size, n_classes) containing the output of the neural
                  network before the softmax layer.
        Returns:
            loss: A 0-d tensor (scalar)
        """
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels_placeholder, logits=preds))
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
        """Perform one step of gradient descent on the provided batch of data.

        Args:
            sess: tf.Session()
            inputs_batch: list of ndarrays (1 per example)
                          each array consists of the glove-indices of the words
                          in the given article-headline pairing
        Returns:
            loss: loss over the batch (a scalar)
        """
        self.sequence_lengths = [len(input_arr) for input_arr in inputs_batch]
        self.max_length = max(sequence_lengths)
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def predict_on_batch(self, sess, inputs_batch):
        """Make predictions for the provided batch of data

        Args:
            sess: tf.Session()
        Returns:
            predictions: np.ndarray of shape (n_samples, n_classes)
        """
        self.sequence_lengths = [len(input_arr) for input_arr in inputs_batch]
        self.max_length = max(sequence_lengths)

        feed = self.create_feed_dict(inputs_batch)
        predictions = sess.run(self.pred, feed_dict=feed)
        preds = tf.argmax(predictions, axis=1).eval()
        return preds

    def run_epoch(self, sess, train_examples, dev_set):
        prog = Progbar(target=1 + train_examples[0].shape[0] / self.config.batch_size)
        for i, (inputs_batch, labels_batch) in enumerate(minibatches(train_examples, self.config.batch_size)):
            loss = self.train_on_batch(sess, inputs_batch, labels_batch)
            prog.update(i + 1, [("train loss", loss)])

        print "Evaluating on dev set"
        actual = vectorize_stances(dev_set[2])
        preds = list(self.predict_on_batch(sess, *dev_set[:2]))
        dev_score = report_score(actual, preds)

        print "- dev Score: {:.2f}".format(dev_score)
        return dev_score

    def fit(self, sess, saver, train_examples, dev_set):
        best_dev_score = 0
        for epoch in range(self.config.n_epochs):
            print "Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs)
            dev_score = self.run_epoch(sess, train_examples, dev_set)
            if dev_score > best_dev_score:
                best_dev_score = dev_score
                if saver:
                    print "New best dev! Saving model in ./data/weights/stance.weights"
                    saver.save(sess, './data/weights/stance.weights')
            print

def main(debug=True):
    if not os.path.exists('./data/weights/'):
        os.makedirs('./data/weights/')

    with tf.Graph().as_default():
        print 80 * "="
        print "INITIALIZING"
        print 80 * "="
        config = Config()

        # Load Data
        # Note: X_train_input, X_dev_input, X_test_input are lists where each item is an example.
        # Each example is a sparse representation of a headline + article, where the text 
        # is encoded as a series of indices into the glove-vectors.
        # y_train_input, y_dev_input, y_test_input are matrices (num_examples, num_classes)
        X_train_input, X_dev_input, X_test_input, y_train_input, y_dev_input, y_test_input = read_binaries()

        # Create Basic LSTM Model
        model = BasicLSTM(config)
        
        # Create Data Lists
        train_examples = [X_train_input, y_train_input]
        dev_set = [X_dev_input, y_dev_input]
        test_set = [X_test_input, y_test_input]

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
            model.fit(session, saver, train_examples, dev_set)

            if not debug:
                print 80 * "="
                print "TESTING"
                print 80 * "="
                print "Restoring the best model weights found on the dev set"
                saver.restore(session, './data/weights/stance.weights')
                print "Final evaluation on test set",

                actual = vectorize_stances(test_set[2])
                preds = list(model.predict_on_batch(session, *test_set[:2]))
                test_score = report_score(actual, preds)

                print "- test Score: {:.2f}".format(test_score)
                print "Writing predictions"
                with open('snli_nn_baseline_test_predicted.pkl', 'w') as f:
                    cPickle.dump(preds, f, -1)
                print "Done!"

if __name__ == '__main__':
    main(False)
