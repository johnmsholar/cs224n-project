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
        self.context_hidden_size = 300 # Hidden State Size
        self.aggregate_hidden_size = 300
        self.squashing_layer_hidden_size = 150
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
        self.classification_problem = 3
        self.max_headline_length = 500
        self.max_article_length = 800
        self.uniform_data_split = False  


class Bimpmp(Advanced_Model):
    """ Conditional Encoding LSTM Model.
    """
    def get_model_name(self):
        return 'bimpmp'

    def get_fn_names(self):
        """ Retrieve file names.
            fn_names = [best_weights_fn, curr_weights_fn, preds_fn]
        """
        best_weights_fn = 'bimpmp_best_stance.weights'
        curr_weights_fn = 'bimpmp_curr_stance.weights'
        preds_fn = 'bimpmp_predicted.pkl'
        best_train_weights_fn = 'bimpmp_best_train_stance.weights'
        return [best_weights_fn, curr_weights_fn, preds_fn, best_train_weights_fn]

    def add_prediction_op(self, debug): 
        """ Runs RNN on the input. 
        """
        # Lookup Glove Embeddings for the headline words (e.g. one at each time step)
        headline_x = self.add_embedding(headline_embedding=True)
        body_x = self.add_embedding(headline_embedding=False)
        dropout_rate = self.dropout_placeholder

        # Context Layer
        with tf.variable_scope("context_layer"):
            # run first headline LSTM
            fw_cell = tf.contrib.rnn.LSTMBlockCell(num_units=self.config.context_hidden_size)
            bw_cell = tf.contrib.rnn.LSTMBlockCell(num_units=self.config.context_hidden_size)           
            headline_context_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                fw_cell,
                bw_cell,
                headline_x,
                dtype=tf.float32,
                sequence_length=self.h_seq_lengths_placeholder
            )
            article_context_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                fw_cell,
                bw_cell,
                body_x,
                dtype=tf.float32,
                sequence_length=self.a_seq_lengths_placeholder
            )           

        # Matching Layer -- assume output is concatenated (fw and bw together)
        # Output Dimensionality: batch_size x time steps x (hidden_size x 2)
        with tf.variable_scope("matching_headline_to_article"):
            matching_layer_headline_to_article = Multiperspective_Matching_A_to_B_Layer()
            post_matching_h_to_a = matching_layer_headline_to_article(headline_context_outputs, article_context_outputs)

        with tf.variable_scope("matching_article_to_headline"):
            matching_layer_article_to_headline = Multiperspective_Matching_A_to_B_Layer()
            post_matching_a_to_h = matching_layer_article_to_headline(article_context_outputs, headline_context_outputs)

        # Aggregation Layer 
        with tf.variable_scope("aggregation_layer"):
            # run first headline LSTM
            fw_cell_aggregation = tf.contrib.rnn.LSTMBlockCell(num_units=self.config.aggregate_hidden_size)
            bw_cell_aggregation = tf.contrib.rnn.LSTMBlockCell(num_units=self.config.aggregate_hidden_size)           
            headline_aggregate_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                fw_cell_aggregation,
                bw_cell_aggregation,
                post_matching_h_to_a,
                dtype=tf.float32,
                sequence_length=self.h_seq_lengths_placeholder
            )
            article_aggregate_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                fw_cell_aggregation,
                bw_cell_aggregation,
                post_matching_a_to_h,
                dtype=tf.float32,
                sequence_length=self.a_seq_lengths_placeholder
            )           

        # Final Prediction Layer
        with tf.variable_scope("final_prediction_layer"):
            headline_output = tf.concat([headline_aggregate_outputs[0][:, -1, :], headline_aggregate_outputs[1][:, -1, :]])
            article_output = tf.concat([article_aggregate_outputs[0][:, -1, :], article_aggregate_outputs[1][:, -1, :]]) 
            output = tf.concat([headline_output, article_output], 1)
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

        return preds, debug_ops

def main(debug=True):
    # Parse Arguments
    arg_epoch, arg_restore = parse_args()

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
                restore_path = '{}/{}'.format(weights_path, model.get_fn_names()[1])
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
                test_score, _, test_confusion_matrix_str = model.predict(session, test_set, save_preds=True)
                with open(model.test_confusion_matrix_fn, 'w') as file:
                    file.write(test_confusion_matrix_str)


if __name__ == '__main__':
    main(False)