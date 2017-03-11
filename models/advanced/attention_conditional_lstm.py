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
from fnc1_utils.featurizer import create_embeddings
from util import create_tensorflow_saver
from layers.attention_layer import AttentionLayer
from layers.class_squash_layer import ClassSquashLayer

class Config:
    """Holds model hyperparams and data information.
    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation. Use self.config.? instead of Config.?
    """
    num_classes = 3 # Number of classses for classification task.
    embed_size = 300 # Size of Glove Vectors

    # Hyper Parameters
    hidden_size = 300 # Hidden State Size
    batch_size = 50
    n_epochs = None
    lr = 0.02
    max_grad_norm = 5.
    dropout_rate = 0.0
    beta = 0

class Attention_Conditonal_Encoding_LSTM_Model(Advanced_Model):
    """ Conditional Encoding LSTM Model.
    """
    def get_fn_names(self):
        """ Retrieve file names.
            fn_names = [best_weights_fn, curr_weights_fn, preds_fn]
        """
        best_weights_fn = 'attention_conditional_lstm_best_stance.weights'
        curr_weights_fn = 'attention_conditional_lstm_curr_stance.weights'
        preds_fn = 'attention_conditional_encoding_lstm_predicted.pkl'
        return [best_weights_fn, curr_weights_fn, preds_fn]

    def add_prediction_op(self): 
        """Runs RNN on the input. 
        """
        # Lookup Glove Embeddings for the headline words (e.g. one at each time step)
        headline_x = self.add_embedding(headline_embedding=True)
        body_x = self.add_embedding(headline_embedding=False)
        dropout_rate = self.dropout_placeholder

        # run first headline LSTM
        with tf.variable_scope("headline_cell"):
            cell_headline = tf.contrib.rnn.LSTMBlockCell(num_units=self.config.hidden_size)
            headline_outputs, headline_state = tf.nn.dynamic_rnn(cell_headline, headline_x, dtype=tf.float32, sequence_length = self.h_seq_lengths_placeholder)

        # run second LSTM that accept state from first LSTM
        with tf.variable_scope("body_cell"):
            cell_body = tf.contrib.rnn.LSTMBlockCell(num_units = self.config.hidden_size)
            outputs, _ = tf.nn.dynamic_rnn(cell_body, body_x, initial_state=headline_state, dtype=tf.float32, sequence_length = self.a_seq_lengths_placeholder)

        h_n = outputs[:,-1,:]

        attention_layer = AttentionLayer(self.config.hidden_size, self.h_max_length)
        output = attention_layer(headline_outputs, h_n)

        # Compute predictions
        # output_dropout = tf.nn.dropout(output, dropout_rate)
        class_squash_layer = ClassSquashLayer(self.config.hidden_size, self.config.num_classes)
        preds = class_squash_layer(output)
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
        training_size=.80,
        random_split=False,
        truncate_headlines=False,
        truncate_articles=True,
        classification_problem=3,
        max_headline_length=500,
        max_article_length=500,
        glove_set=None,
    )   

    X, y = produce_uniform_data_split(X, y)

    # Each set is of the form:
    # [headline_glove_index_matrix, article_glove_index_matrix, h_seq_lengths, a_seq_lengths, labels]
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
        model = Attention_Conditonal_Encoding_LSTM_Model(config, report_score, max_input_lengths, glove_matrix)
        model.print_params()
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
                test_score, _ = model.predict(session, test_set, save_preds=True)
                print "- test Score: {:.2f}".format(test_score)

if __name__ == '__main__':
    main(False)
