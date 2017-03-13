#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
CS 224N 2016-2017 
two_lstm_encoders.py: 
    Leverage 2 different LSTMS to encode the article
    and the headline into two different representations
    of the same dimensionality. Then concanetate these 
    two representations and feed through an MLP to produce
    a classification.
Sahil Chopra <schopra8@cs.stanford.edu>
Saachi Jain <saachi@cs.stanford.edu>
John Sholar <jmsholar@cs.stanford.edu>

NOTE: Potentially performing redundant work and overtraining the
      article LSTM by feeding the same articles multiple times into
      the LSTM during training.
"""

import argparse
import tensorflow as tf
import numpy as np

import time
import os
import sys
sys.path.insert(0, '../')

from advanced_model import Advanced_Model, create_data_sets_for_model
from fnc1_utils.score import report_score
from fnc1_utils.featurizer import create_inputs_by_glove
from util import create_tensorflow_saver

class Config(object):
    """Holds model hyperparams and data information.
    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation. Use self.config.? instead of Config.?
    """
    def __init__(self):
        self.num_classes = 3 # Number of classses for classification task.
        self.embed_size = 300 # Size of Glove Vectors

        # Hyper Parameters
        self.hidden_size = 300 # Hidden State Size
        self.batch_size = 50
        self.n_epochs = None
        self.lr = 0.001
        self.max_grad_norm = 5.
        self.dropout_rate = 1.0
        self.beta = 0

class Two_LSTM_Encoders_Model(Advanced_Model):
    """ 1 LSTM to encode headline.
        1 LSTM to encode article.
        Concatenate final hidden representations and feed through and MLP
        to determine final prediction.
    """

    def get_model_name(self):
        return 'two_lstm_encoders'

    def get_fn_names(self):
        """ Retrieve file names.
            fn_names = [best_weights_fn, curr_weights_fn, preds_fn, best_train_weights_fn]
        """
        best_weights_fn = 'two_lstm_encoders_best_stance.weights'
        curr_weights_fn = 'two_lstm_encoders_curr_stance.weights'
        preds_fn = 'two_lstm_encoders_predicted.pkl'
        best_train_weights_fn = 'two_lstm_encoders_best_train_stance.weights'
        return [best_weights_fn, curr_weights_fn, preds_fn, best_train_weights_fn]

    def add_prediction_op(self): 
        """ 1 LSTM on headlines.
            1 LSTM on articles.
            Concatenate final outputs of both LSTMs and run though a 1 Layer MLP.

        Returns:
            preds: tf.Tensor of shape (batch_size, self.config.num_classes)
        """
        # Lookup Glove Embeddings for the input words (e.g. one at each time step)
        headlines_x = self.add_embedding()
        articles_x = self.add_embedding(False)
        dropout_rate = self.dropout_placeholder

        # Create final layer to project the output from th RNN onto
        # the four classification labels.
        U = tf.get_variable("U", shape=[self.config.hidden_size * 2, self.config.num_classes],
            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("b", shape=[self.config.num_classes],
            initializer=tf.constant_initializer(0))

        # Headlines -- Compute the output at the end of the LSTM (automatically unrolled)
        with tf.variable_scope("headline_cell"):
            headline_cell = tf.contrib.rnn.LSTMBlockCell(num_units=self.config.hidden_size)
            headline_outputs, _ = tf.nn.dynamic_rnn(headline_cell, headlines_x, dtype=tf.float32, sequence_length = self.h_seq_lengths_placeholder)
            headline_output = headline_outputs[:, -1, :]
            assert headline_output.get_shape().as_list() == [None, self.config.hidden_size], "predictions are not of the right shape. Expected {}, got {}".format([None, self.config.hidden_size], headline_output.get_shape().as_list())

        # Articles -- Compute the output at the end of the LSTM (automatically unrolled)
        with tf.variable_scope("article_cell"):
            article_cell = tf.contrib.rnn.LSTMBlockCell(num_units=self.config.hidden_size)
            article_outputs, _ = tf.nn.dynamic_rnn(article_cell, articles_x, dtype=tf.float32, sequence_length = self.a_seq_lengths_placeholder)
            article_output = article_outputs[:, -1, :]
            assert article_output.get_shape().as_list() == [None, self.config.hidden_size], "predictions are not of the right shape. Expected {}, got {}".format([None, self.config.hidden_size], article_output.get_shape().as_list())

        # Compute dropout on both headlines and articles
        headline_output_dropout = tf.nn.dropout(headline_output, dropout_rate)
        article_output_dropout = tf.nn.dropout(article_output, dropout_rate)

        # Concatenate headline and article outputs
        output = tf.concat([headline_output_dropout, article_output_dropout], 1)
        preds = tf.matmul(output, U) + b
        assert preds.get_shape().as_list() == [None, self.config.num_classes], "predictions are not of the right shape. Expected {}, got {}".format([None, self.config.num_classes], preds.get_shape().as_list())
        return preds

def main(debug=True):
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--restore', action='store_true')
    args = parser.parse_args()

    # Create Config
    config = Config()
    if args.epoch:
        config.n_epochs = args.epoch

    # Load Data
    X, y, glove_matrix, max_input_lengths, word_to_glove_index = create_embeddings(
        training_size=0.8,
        random_split=False,
        truncate_headlines=False,
        truncate_articles=True,
        classification_problem=3,
        max_headline_length=500,
        max_article_length=500,
        glove_set=None,
        debug=debug
    )   

    # TODO: Remove This
    X, y = produce_uniform_data_split(X, y)

    with tf.Graph().as_default():
        print 80 * "="
        print "INITIALIZING"
        print 80 * "="

        # Create and configure model
        print "Building model...",
        model = Two_LSTM_Encoders_Model(config, report_score, max_lengths, glove_matrix)
        start = time.time()
        print "took {:.2f} seconds\n".format(time.time() - start)

        # Initialize variables
        init = tf.global_variables_initializer()
        saver = None if debug else tf.train.Saver()

        with tf.Session() as session:
            # Load weights if necessary
            session.run(init)
            saver = create_tensorflow_saver(model.exclude_names)
            if args.restore:
                saver.restore(session, model.curr_weights_fn)

            # Finalize graph
            session.graph.finalize()

            # Train Model
            print 80 * "="
            print "TRAINING"
            print 80 * "="
            model.fit(session, saver, train_examples, dev_set)

            if not debug:
                print 80 * "="
                print "TESTING"
                print 80 * "="
                print "Restoring the best model weights found on the dev set"
                saver.restore(session, model.best_weights_fn)

                print "Final evaluation on test set",
                test_score, _, _= model.predict(session, test_set, save_preds=True)
                print "- test Score: {:.2f}".format(test_score)
if __name__ == '__main__':
    main(False)
