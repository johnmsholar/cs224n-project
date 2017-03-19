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
from util import create_tensorflow_saver, parse_args
from layers.attention_layer import AttentionLayer
from layers.class_squash_layer import ClassSquashLayer

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
        self.squashing_layer_hidden_size = 150
        self.batch_size = 50
        self.n_epochs = None
        self.lr = 0.0001
        self.max_grad_norm = 5.
        self.dropout_rate = 0.9
        self.beta = 0.01

        # Data Params
        self.training_size = .80
        self.random_split = False
        self.truncate_headlines = False
        self.truncate_articles = True
        self.classification_problem = 3
        self.max_headline_length = 500
        self.max_article_length = 800
        self.uniform_data_split = False  


class Attention_Conditonal_Encoding_LSTM_Model(Advanced_Model):
    """ Conditional Encoding LSTM Model.
    """
    def get_model_name(self):
        return 'attention_conditional_lstm'

    def get_fn_names(self):
        """ Retrieve file names.
            fn_names = [best_weights_fn, curr_weights_fn, preds_fn]
        """
        best_weights_fn = 'attention_conditional_lstm_best_stance.weights'
        curr_weights_fn = 'attention_conditional_lstm_curr_stance.weights'
        preds_fn = 'attention_conditional_encoding_lstm_predicted.pkl'
        best_train_weights_fn = 'attention_conditional_encoding_lstm_best_train_stance.weights'
        return [best_weights_fn, curr_weights_fn, preds_fn, best_train_weights_fn]

    def add_prediction_op(self, debug): 
        """ Runs RNN on the input. 
        """
        # Lookup Glove Embeddings for the headline words (e.g. one at each time step)
        headline_x = self.add_embedding(headline_embedding=True)
        body_x = self.add_embedding(headline_embedding=False)
        dropout_rate = self.dropout_placeholder

        # run first headline LSTM
        # headline_x_list = [headline_x[:, i, :] for i in range(headline_x.get_shape()[1].value)]
        with tf.variable_scope("headline_cell"):
            cell_headline = tf.contrib.rnn.LSTMBlockCell(num_units=self.config.hidden_size)
            # headline_outputs, headline_state = tf.contrib.rnn.static_rnn(cell_headline, headline_x_list, dtype=tf.float32)
            headline_outputs, headline_state = tf.nn.dynamic_rnn(cell_headline, headline_x, dtype=tf.float32, sequence_length = self.h_seq_lengths_placeholder)

        # run second LSTM that accept state from first LSTM
        # body_x_list = [body_x[:, i, :] for i in range(body_x.get_shape()[1].value)]
        with tf.variable_scope("body_cell"):
            cell_body = tf.contrib.rnn.LSTMBlockCell(num_units = self.config.hidden_size)
            # _, article_state = tf.contrib.rnn.static_rnn(cell_body, body_x_list, initial_state=headline_state, dtype=tf.float32)
            outputs, article_state = tf.nn.dynamic_rnn(cell_body, body_x, initial_state=headline_state, dtype=tf.float32, sequence_length = self.a_seq_lengths_placeholder)
        
        # Apply attention
        with tf.variable_scope("attention"):
            attention_layer = AttentionLayer(self.config.hidden_size, self.h_max_length)
            output = attention_layer(headline_outputs, article_state[1])

        # Compute predictions
        with tf.variable_scope("final_projection"):
            output_dropout = tf.nn.dropout(output, dropout_rate)
            squash = tf.contrib.layers.fully_connected(
                    inputs=output_dropout,
                    num_outputs=self.config.squashing_layer_hidden_size,
                    activation_fn=tf.nn.relu,
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    biases_initializer=tf.constant_initializer(0),
            )
            preds = tf.contrib.layers.fully_connected(
                    inputs=squash,
                    num_outputs=self.config.num_classes,
                    activation_fn=tf.nn.relu,
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    biases_initializer=tf.constant_initializer(0),
            )

        # Debugging Ops
        if debug:
            headline_x = tf.Print(headline_x, [headline_x], 'headline_x', summarize=20)
            body_x = tf.Print(body_x, [body_x], 'body_x', summarize=24)
            h_seq_lengths = tf.Print(self.h_seq_lengths_placeholder, [self.h_seq_lengths_placeholder], 'h_seq_lengths', summarize=3)
            a_seq_lengths = tf.Print(self.a_seq_lengths_placeholder, [self.a_seq_lengths_placeholder], 'a_seq_lengths', summarize=3)            
            headline_outputs = tf.Print(headline_outputs, [headline_outputs], 'headline_outputs', summarize=20)
            headline_state = tf.Print(headline_state, [headline_state], 'headline_state', summarize=20)
            article_state = tf.Print(article_state, [article_state], 'article_state', summarize=20)
            debug_ops = [headline_x, body_x, h_seq_lengths, a_seq_lengths, headline_outputs, headline_state, article_state]
        else:
            debug_ops = None

        return preds, debug_ops

def main(debug=True):
    # Parse Arguments
    arg_epoch, arg_restore, arg_test = parse_args()

    # Create Config
    config = Config()
    if arg_epoch:
        config.n_epochs = arg_epoch

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
        start = time.time()
        model = Attention_Conditonal_Encoding_LSTM_Model(config, report_score, max_input_lengths, glove_matrix, debug)
        model.print_params()
        print "took {:.2f} seconds\n".format(time.time() - start)

        # Initialize variables
        init = tf.global_variables_initializer()
        saver = None if debug else tf.train.Saver()

        with tf.Session() as session:
            # Load weights if necessary
            session.run(init)
            saver = create_tensorflow_saver(model.exclude_names)
            if arg_restore != None:
                weights_path = './data/{}/{}/weights'.format(model.get_model_name(), arg_restore)
                restore_path = '{}/{}'.format(weights_path, model.get_fn_names()[0])
                saver.restore(session, model.curr_weights_fn)

            # Finalize graph
            session.graph.finalize()

            if not arg_test:
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
                # TODO: Remove this hack
                model.best_weights_fn = 'data/attention_conditional_lstm/1489883343.29/weights/attention_conditional_lstm_best_stance.weights'
                saver.restore(session, model.best_weights_fn)

                print "Final evaluation on test set",
                test_score, _, test_confusion_matrix_str = model.predict(session, test_set, save_preds=True)
                with open(model.test_confusion_matrix_fn, 'w') as file:
                    file.write(test_confusion_matrix_str)


if __name__ == '__main__':
    main(False)
