#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
CS 224N 2016-2017 
conditional_lstm.py: Conditional LSTM Implementation
Sahil Chopra <schopra8@cs.stanford.edu>
Saachi Jain <saachi@cs.stanford.edu>
John Sholar <jmsholar@cs.stanford.edu>
"""

import argparse
import tensorflow as tf
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

from fnc1_utils.score import report_score, pretty_report_score
from model import Model
from fnc1_utils.featurizer import create_inputs_by_glove
from util import Progbar, vectorize_stances, minibatches, create_tensorflow_saver

class Config:
    """Holds model hyperparams and data information.
    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation. Use self.config.? instead of Config.?
    """
    num_classes = 4 # Number of classses for classification task.
    embed_size = 300 # Size of Glove Vectors

    h_max_length = None # set when configuring inputs, headline max length
    b_max_length = None # set when configuring inputs, body max length

    # Hyper Parameters
 
    hidden_size = 300 # Hidden State Size
    batch_size = 50
    n_epochs = None
    lr = 0.02
    max_grad_norm = 5.
    dropout_rate = 0.5
    beta = 0.2

    # Other params
    pretrained_embeddings = None

class Conditonal_Encoding_LSTM_Model(Model):
    """
    """

    def __init__(self, config):
        self.config = config

        # Defining placeholders.
        self.h_sequence_lengths_placeholder = None
        self.headline_placeholder = None

        self.b_sequence_lengths_placeholder = None
        self.body_placeholder = None

        self.labels_placeholder = None
        self.dropout_placeholder = None
        self.embedding_matrix = tf.constant(self.config.pretrained_embeddings, dtype=tf.float32, name="embeddings_matrix")

        self.build()
        self.argmax = tf.argmax(self.pred, axis=1)

    def add_placeholders(self):
        """ Generates placeholder variables to represent the input tensors.
        """
        # Sequence Lengths
        self.h_sequence_lengths_placeholder = tf.placeholder(tf.int32, (None), name="h_seq_lengths")
        self.b_sequence_lengths_placeholder = tf.placeholder(tf.int32, (None), name="b_seq_lengths")

        #inputs
        self.headline_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.h_max_length), name="headline_input")
        self.body_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.b_max_length), name="body_input")

        self.labels_placeholder = tf.placeholder(tf.float32, (None, self.config.num_classes), name="labels")
        self.dropout_placeholder = tf.placeholder(tf.float32, name="dropout")

    def create_feed_dict(self, headline_batch, body_batch, labels_batch=None, dropout=1, h_sequence_lengths=[], b_sequence_lengths=[]):
        """Creates the feed_dict for the model.
        """
        feed_dict = {
            self.headline_placeholder: headline_batch,
            self.body_placeholder: body_batch,
            self.dropout_placeholder: dropout,
            self.h_sequence_lengths_placeholder: h_sequence_lengths,
            self.b_sequence_lengths_placeholder: b_sequence_lengths,
        }
        if labels_batch is not None:
            feed_dict[self.labels_placeholder]= labels_batch
        return feed_dict

    def add_embedding(self, isHeadline):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:
        Parms:
            isHeadline: true if we want embeddings for headline, false if we want embeddings for body
        Returns:
            embeddings: tf.Tensor of shape (None, max_length, embed_size)
        """
        if isHeadline:
            e = tf.nn.embedding_lookup(self.embedding_matrix, self.headline_placeholder)
            embeddings = tf.reshape(e, shape=[-1, self.config.h_max_length, self.config.embed_size], name="embeddings_matrix_h")
        else:
            e = tf.nn.embedding_lookup(self.embedding_matrix, self.body_placeholder)
            embeddings = tf.reshape(e, shape=[-1, self.config.b_max_length, self.config.embed_size], name="embeddings_matrix_b")
        end_time = time.time()
        return embeddings

    def add_prediction_op(self): 
        """Runs RNN on the input. 

        Returns:
            preds: tf.Tensor of shape (batch_size, self.config.hidden_size)
        """
        # Lookup Glove Embeddings for the headline words (e.g. one at each time step)
        headline_x = self.add_embedding(True)
        body_x = self.add_embedding(False)
        dropout_rate = self.dropout_placeholder

        # Create final layer to project the output from th RNN onto
        # the four classification labels.
        U = tf.get_variable("U", shape=[self.config.hidden_size, self.config.num_classes],
            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("b", shape=[self.config.num_classes],
            initializer=tf.constant_initializer(0))

        # run first headline LSTM
        with tf.variable_scope("headline_cell"):
            cell_headline = tf.contrib.rnn.LSTMBlockCell(num_units=self.config.hidden_size)
            _, headline_state = tf.nn.dynamic_rnn(cell_headline, headline_x, dtype=tf.float32, sequence_length = self.h_sequence_lengths_placeholder)

        # run second LSTM that accept state from first LSTM
        with tf.variable_scope("body_cell"):
            cell_body = tf.contrib.rnn.LSTMBlockCell(num_units = self.config.hidden_size)
            outputs, _ = tf.nn.dynamic_rnn(cell_body, body_x, initial_state=headline_state, dtype=tf.float32, sequence_length = self.b_sequence_lengths_placeholder)

        output = outputs[:,-1,:]
        assert output.get_shape().as_list() == [None, self.config.hidden_size], "predictions are not of the right shape. Expected {}, got {}".format([None, self.config.hidden_size], output.get_shape().as_list())

        # Compute predictions
        output_dropout = tf.nn.dropout(output, dropout_rate)
        preds = tf.matmul(output_dropout, U) + b
        assert preds.get_shape().as_list() == [None, self.config.num_classes], "predictions are not of the right shape. Expected {}, got {}".format([None, self.config.num_classes], preds.get_shape().as_list())
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
        reg = 0
        for var in tf.trainable_variables():
            reg += tf.reduce_mean(tf.nn.l2_loss(var))
        reg *= self.config.beta
        loss += reg
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


    def train_on_batch(self, sess, headline_batch, body_batch, labels_batch):
        """Perform one step of gradient descent on the provided batch of data.

        Args:
            sess: tf.Session()
            inputs_batch: list of ndarrays (1 per example)
                          each array consists of the glove-indices of the words
                          in the given article-headline pairing
        Returns:
            loss: loss over the batch (a scalar)
        """
        h_sequence_lengths = [len(input_arr) for input_arr in headline_batch]
        b_sequence_lengths = [len(input_arr) for input_arr in body_batch]

        feed = self.create_feed_dict(headline_batch, body_batch, labels_batch=labels_batch, h_sequence_lengths=h_sequence_lengths, b_sequence_lengths = b_sequence_lengths, dropout = self.config.dropout_rate)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def predict_on_batch(self, sess, headline_batch, body_batch):
        """Make predictions for the provided batch of data

        Args:
            sess: tf.Session()
        Returns:
            predictions: np.ndarray of shape (n_samples, n_classes)
        """
        h_sequence_lengths = [len(input_arr) for input_arr in headline_batch]
        b_sequence_lengths = [len(input_arr) for input_arr in body_batch]
        feed = self.create_feed_dict(headline_batch, body_batch, h_sequence_lengths=h_sequence_lengths, b_sequence_lengths = b_sequence_lengths)
        predictions = sess.run(self.argmax, feed_dict=feed)
        return predictions

    # train_examples should be (headline_matrix, body_matrix, labels_match)
    def run_epoch(self, sess, train_examples, dev_set):
        prog = Progbar(target=1 + len(train_examples[0])/ self.config.batch_size)
        for i, (headline_batch, article_batch, labels_batch) in enumerate(minibatches(train_examples, self.config.batch_size)):
            loss = self.train_on_batch(sess, headline_batch, article_batch, labels_batch)
            prog.update(i + 1, [("train loss", loss)])

        print "Evaluating on dev set"
        actual = vectorize_stances(dev_set[2])
        preds = []
        for i, (headline_batch, article_batch, labels_batch) in enumerate(minibatches(dev_set, self.config.batch_size)):
            predictions_batch = list(self.predict_on_batch(sess, headline_batch, article_batch))
            preds.extend(predictions_batch)
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
                    print "New best dev! Saving model in ./data/weights/conditional_lstm_best_stance.weights"
                    saver.save(sess, './data/weights/conditional_lstm_best_stance.weights')
            if saver:
                print "Finished Epoch ... Saving model in ./data/weights/conditional_lstm_curr_stance.weights"
                saver.save(sess, './data/weights/conditional_lstm_curr_stance.weights')
            print

def main(debug=True):

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--restore', action='store_true')
    args = parser.parse_args()

    if not os.path.exists('./data/weights/'):
        os.makedirs('./data/weights/')

    if not os.path.exists('./data/predictions/'):
        os.makedirs('./data/predictions/')

    if not os.path.exists('./data/plots/'):
        os.makedirs('./data/plots/')

    with tf.Graph().as_default():
        print 80 * "="
        print "INITIALIZING"
        print 80 * "="
        config = Config()
        if args.epoch:
            config.n_epochs = args.epoch

        # Load Data
        # Note: X_train_input, X_dev_input, X_test_input are lists where each item is an example.
        # Each example is a dense representation of a (headline, article), where the text 
        # is encoded as a series of indices into the glove-vectors.
        # y_train_input, y_dev_input, y_test_input are matrices (num_examples, num_classes)
        X_train_input, X_dev_input, X_test_input, y_train_input, y_dev_input, y_test_input, glove_matrix, max_lengths= create_inputs_by_glove(concatenate=False)
        config.h_max_length = max_lengths[0]
        config.b_max_length = max_lengths[1]
        # Create Conditional Encoding LSTM Model
        config.pretrained_embeddings = glove_matrix
        model = Conditonal_Encoding_LSTM_Model(config)
        
        # Create Data Lists
        train_examples = [X_train_input[0], X_train_input[1], y_train_input]
        dev_set = [X_dev_input[0], X_dev_input[1], y_dev_input]
        test_set = [X_test_input[0], X_test_input[1], y_test_input]
        print "Building model...",
        start = time.time()
        print "took {:.2f} seconds\n".format(time.time() - start)

        init = tf.global_variables_initializer()
        saver = None if debug else tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            exclude_names = set(["embeddings_matrix:0", "embeddings_matrix_h:0", "embeddings_matrix_b:0"])
            saver = create_tensorflow_saver(exclude_names)
            if args.restore:
                saver.restore(session, './data/weights/conditional_lstm_curr_stance.weights')
            session.graph.finalize()

            print 80 * "="
            print "TRAINING"
            print 80 * "="
            model.fit(session, saver, train_examples, dev_set)

            if not debug:
                print 80 * "="
                print "TESTING"
                print 80 * "="
                print "Restoring the best model weights found on the dev set"
                saver.restore(session, './data/weights/conditional_lstm_best_stance.weights')

                print "Final evaluation on test set",
                actual = vectorize_stances(test_set[2])
                preds = []
                for i, (headline_batch, article_batch, labels_batch) in enumerate(minibatches(test_set, config.batch_size)):
                    predictions_batch = list(model.predict_on_batch(sess, headline_batch, article_batch))
                    preds.extend(predictions_batch)
                test_score = pretty_report_score(actual, preds, "./data/plots/conditional_lstm_confusion_matrix.png")
                print "- test Score: {:.2f}".format(test_score)
                print "Writing predictions"
                with open('./data/predictions/conditional_encoding_lstm_predicted.pkl', 'w') as f:
                    cPickle.dump(preds, f, -1)
                print "Done!"

if __name__ == '__main__':
    main(False)
