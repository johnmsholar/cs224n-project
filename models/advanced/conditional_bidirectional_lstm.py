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

import time
import os
import sys
sys.path.insert(0, '../')

from advanced_model import Advanced_Model, create_data_sets_for_model, produce_uniform_data_split
from fnc1_utils.score import report_score
from fnc1_utils.featurizer import create_inputs_by_glove, create_embeddings
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
        self.lr = 0.0001
        self.max_grad_norm = 5.
        self.dropout_rate = 0.8
        self.beta = 0

        # Data Params
        self.training_size = .80
        self.random_split = False
        self.truncate_headlines = False
        self.truncate_articles = True
        self.classification_problem = 1
        self.max_headline_length = 500
        self.max_article_length = 800
        self.uniform_data_split = False  
        
class Conditional_Encoding_Bidirectional_LSTM_Model(Advanced_Model):
    """ Bidirectional Conditional Encoding LSTM Model.
    """
    def get_model_name(self):
        return 'bidirectional_conditional_lstm'

    def get_fn_names(self):
        """ Retrieve file names.
            fn_names = [best_weights_fn, curr_weights_fn, preds_fn, best_train_weights_fn]
        """
        best_weights_fn = 'Conditional_Encoding_Bidirectional_LSTM_Model_best_stance.weights'
        curr_weights_fn = 'Conditional_Encoding_Bidirectional_LSTM_Model_curr_stance.weights'
        preds_fn = 'Conditional_Encoding_Bidirectional_LSTM_Model_predicted.pkl'
        best_train_weights_fn = 'Conditional_Encoding_Bidirectional_LSTM_Model_predicted_best_train_stance.weights'
        return [best_weights_fn, curr_weights_fn, preds_fn, best_train_weights_fn]

    def add_prediction_op(self, debug):
        """Runs RNN on the input. 
        """
        # Lookup Glove Embeddings for the headline words (e.g. one at each time step)
        headline_x = self.add_embedding(headline_embedding=True)
        body_x = self.add_embedding(headline_embedding=False)
        dropout_rate = self.dropout_placeholder

        with tf.variable_scope("headline_cell"):
            # run first headline LSTM
            headline_fw_cell = tf.contrib.rnn.LSTMBlockCell(num_units=self.config.hidden_size)
            headline_bw_cell = tf.contrib.rnn.LSTMBlockCell(num_units=self.config.hidden_size)           
            headline_outputs, headline_states = tf.nn.bidirectional_dynamic_rnn(
                headline_fw_cell,
                headline_bw_cell,
                headline_x,
                dtype=tf.float32,
                sequence_length=self.h_seq_lengths_placeholder
            )

        with tf.variable_scope("article_cell"):
            article_fw_cell = tf.contrib.rnn.LSTMBlockCell(num_units=self.config.hidden_size)
            article_bw_cell = tf.contrib.rnn.LSTMBlockCell(num_units=self.config.hidden_size)
            outputs, article_states = tf.nn.bidirectional_dynamic_rnn(
                article_fw_cell,
                article_bw_cell,
                body_x,
                initial_state_fw=headline_states[0],
                initial_state_bw=headline_states[1],
                dtype=tf.float32,
                sequence_length= self.a_seq_lengths_placeholder
            )
            fw_output = article_states[0][1]
            bw_output = outputs[1][1]
            output = tf.concat([fw_output, bw_output], 1)
            output_dropout = tf.nn.dropout(output, dropout_rate)
            assert output_dropout.get_shape().as_list() == [None, self.config.hidden_size * 2], "predictions are not of the right shape. Expected {}, got {}".format([None, self.config.hidden_size * 2], output_dropout.get_shape().as_list())

        with tf.variable_scope("final_projection"):
            # Compute predictions
            preds = tf.contrib.layers.fully_connected(
                inputs=output_dropout,
                num_outputs=self.config.num_classes,
                activation_fn=tf.nn.relu,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                biases_initializer=tf.constant_initializer(0),
            )
            assert preds.get_shape().as_list() == [None, self.config.num_classes], "predictions are not of the right shape. Expected {}, got {}".format([None, self.config.num_classes], preds.get_shape().as_list())

        # Debugging Ops
        if debug:
            headline_x = tf.Print(headline_x, [headline_x], 'headline_x',summarize=20)
            body_x = tf.Print(body_x, [body_x], 'body_x', summarize=24)
            h_seq_lengths = tf.Print(self.h_seq_lengths_placeholder, [self.h_seq_lengths_placeholder],'h_seq_lengths', summarize=3)
            a_seq_lengths = tf.Print(self.a_seq_lengths_placeholder, [self.a_seq_lengths_placeholder],'a_seq_lengths', summarize=3)
            debug_ops = [headline_x, body_x, h_seq_lengths, a_seq_lengths]
        else:
            debug_ops = None

        return preds, debug_ops

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
        training_size=config.training_size,
        random_split=config.random_split,
        truncate_headlines=config.truncate_headlines,
        truncate_articles=config.truncate_articles,
        classification_problem=config.classification_problem,
        max_headline_length=config.max_headline_length,
        max_article_length=config.max_article_length,
        glove_set=None,
        debug=debug
    )   

    if config.uniform_data_split:
        X, y = produce_uniform_data_split(X, y)

    train_examples, dev_set, test_set = create_data_sets_for_model(X, y)
    print "Distribution of Train {}".format(np.sum(train_examples[4], axis=0))
    print "Distribtion of Dev {}".format(np.sum(dev_set[4], axis=0))
    print "Distribution of Test{}".format(np.sum(test_set[4], axis=0))

    with tf.Graph().as_default():
        print 80 * "="
        print "INITIALIZING"
        print 80 * "="

        # Create and configure model
        print "Building model...",
        start = time.time()
        model = Conditional_Encoding_Bidirectional_LSTM_Model(config, report_score, max_input_lengths, glove_matrix, debug=debug)
        model.print_params()
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
                test_score, _, _ = model.predict(session, test_set, save_preds=True)
                print "- test Score: {:.2f}".format(test_score)

if __name__ == '__main__':
    main(False)
