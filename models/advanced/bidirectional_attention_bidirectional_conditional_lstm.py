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

import csv
import time
import os
import sys
sys.path.insert(0, '../')

from advanced_model import Advanced_Model, create_data_sets_for_model, produce_uniform_data_split
from fnc1_utils.score import report_score
from fnc1_utils.featurizer import create_embeddings
from util import create_tensorflow_saver
from layers.attention_layer import AttentionLayer

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
        self.dropout_rate = 0.8
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


class Bidirectional_Attention_Conditonal_Encoding_LSTM_Model(Advanced_Model):
    """ Conditional Encoding LSTM Model.
    """
    def get_model_name(self):
        return 'bidirectional_attention_bidirectional_conditional_lstm'

    def get_fn_names(self):
        """ Retrieve file names.
            fn_names = [best_weights_fn, curr_weights_fn, preds_fn]
        """
        best_weights_fn = 'bidirectional_attention_bidirectional_conditional_lstm_best_stance.weights'
        curr_weights_fn = 'bidirectional_attention_bidirectional_conditional_lstm_curr_stance.weights'
        preds_fn = 'bidirectional_attention_bidirectional_conditional_encoding_lstm_predicted.pkl'
        best_train_weights_fn = 'bidirectional_attention_bidirectional_conditional_encoding_lstm_best_train_stance.weights'
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
            headline_fw_cell = tf.contrib.rnn.LSTMBlockCell(num_units=self.config.hidden_size)
            headline_bw_cell = tf.contrib.rnn.LSTMBlockCell(num_units=self.config.hidden_size)           
            headline_outputs, headline_states = tf.nn.bidirectional_dynamic_rnn(
                headline_fw_cell,
                headline_bw_cell,
                headline_x,
                dtype=tf.float32,
                sequence_length = self.h_seq_lengths_placeholder
            )

        # run second LSTM that accept state from first LSTM
        # body_x_list = [body_x[:, i, :] for i in range(body_x.get_shape()[1].value)]
        with tf.variable_scope("article_cell"):
            article_fw_cell = tf.contrib.rnn.LSTMBlockCell(num_units=self.config.hidden_size)
            article_bw_cell = tf.contrib.rnn.LSTMBlockCell(num_units=self.config.hidden_size)
            article_outputs, article_state = tf.nn.bidirectional_dynamic_rnn(
                article_fw_cell,
                article_bw_cell,
                body_x,
                initial_state_fw=headline_states[0],
                initial_state_bw=headline_states[1],
                dtype=tf.float32,
                sequence_length= self.a_seq_lengths_placeholder
            )

        # Apply attention from headline -> article
        with tf.variable_scope("headline_to_article_attention_fw"):
            article_output = article_state[0][1]
            attention_layer_1 = AttentionLayer(self.config.hidden_size, self.h_max_length)
            output_1 = attention_layer_1(headline_outputs[0], article_output)

        with tf.variable_scope("headline_to_article_attention_bw"):
            article_output = article_state[1][1] 
            attention_layer_2 = AttentionLayer(self.config.hidden_size, self.h_max_length)
            output_2 = attention_layer_2(headline_outputs[1], article_output)

        # Apply attentin from article -> headline
        with tf.variable_scope("article_to_headline_attention_f"):
            headline_output = headline_states[0][1]
            attention_layer_3 = AttentionLayer(self.config.hidden_size, self.a_max_length)
            output_3 = attention_layer_3(article_outputs[0], headline_output)

        with tf.variable_scope("article_to_headline_attention_bw"):
            headline_output = headline_states[1][1]
            attention_layer_4 = AttentionLayer(self.config.hidden_size, self.a_max_length)
            output_4 = attention_layer_4(article_outputs[1], headline_output)

        # Compute predictions
        with tf.variable_scope("final_projection"):
            output = tf.concat([output_1, output_2, output_3, output_4], 1)
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
        model = Bidirectional_Attention_Conditonal_Encoding_LSTM_Model(config, report_score, max_input_lengths, glove_matrix, debug)
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
                test_score, _, test_confusion_matrix_str = model.predict(session, test_set, save_preds=True)
                with open(model.test_confusion_matrix_fn, 'w') as file:
                    file.write(test_confusion_matrix_str)

            # Print Train and Score Files
            with open(model.scores_fn, 'w') as file:
                writer = csv.writer(file)
                writer.writerow(['Epoch Number', "Train Score", "Dev Score"])
                for i, train_value in enumerate(model.train_scores):
                    writer.writerow([i, train_value/100.0, model.dev_scores[i]/100.0])

if __name__ == '__main__':
    main(False)
