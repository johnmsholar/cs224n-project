#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
CS 224N 2016-2017 
classify.py: Classification Harness
Sahil Chopra <schopra8@cs.stanford.edu>
Saachi Jain <saachi@cs.stanford.edu>
John Sholar <jmsholar@cs.stanford.edu>
"""

import tensorflow as tf
import os
import argparse
import sys
import numpy as np
sys.path.insert(0, '../')

from advanced_model import create_data_sets_for_model
from util import create_tensorflow_saver
from fnc1_utils.featurizer import create_embeddings
from advanced.bidirectional_attention_bidirectional_conditional_lstm import Bidirectional_Attention_Conditonal_Encoding_LSTM_Model
from advanced.bimpmp import Bimpmp
from advanced.linear_related_unrelated import classify_related_unrelated
class BimpmpConfig(object):
    """Holds model hyperparams and data information.
    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation. Use self.config.? instead of Config.?
    """
    def __init__(self):
        self.num_classes = 3 # Number of classses for classification task.
        self.embed_size = 100 # Size of Glove Vectors

        # Hyper Parameters
        self.context_hidden_size = 100 # Hidden State Size
        self.aggregate_hidden_size = 100
        self.squashing_layer_hidden_size = 50
        self.batch_size = 30
        self.n_epochs = None
        self.lr = 0.0001
        self.max_grad_norm = 5.
        self.dropout_rate = 0.90
        self.beta = 0.01
        self.num_perspectives = 5

        # Data Params
        self.training_size = .80
        self.random_split = False
        self.truncate_headlines = False
        self.truncate_articles = True
        self.classification_problem = 3
        self.max_headline_length = 500
        self.max_article_length = 800
        self.uniform_data_split = False  

class BiDirAttnBidirCondConfig(object):
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

def create_sub_class_test_data(related):
    '''
        Args: related: a numpy array of booleans that are true if related, false if unrelated
    '''
    unrelated = np.logical_not(related_unrelated_set)
    # includes unrelated
    X, y, glove_matrix, max_input_lengths, word_to_glove_index = create_embeddings(
        training_size=config.training_size,
        random_split=config.random_split,
        truncate_headlines=config.truncate_headlines,
        truncate_articles=config.truncate_articles,
        classification_problem=1,
        max_headline_length=config.max_headline_length,
        max_article_length=config.max_article_length,
        glove_set=None,
        debug=debug
    )
    # isolate test data for classification problem 2
    _, _, (h_glove_index_matrix, a_glove_index_matrix, h_seq_lengths, a_seq_lengths, labels) = create_data_sets_for_model(X, y)

    unrelated_labels = unrelated[labels]
    related_labels = related[labels]

    related_h_glove_index_matrix = related[h_glove_index_matrix]
    related_a_glove_index_matrix = related[a_glove_index_matrix]
    related_h_seq_lengths = np.transpose(related[np.transpose(h_seq_lengths)])
    related_a_seq_lengths = np.transpose(related[np.transpose(a_seq_lengths)])

    return glove_matrix, related_h_glove_index_matrix, related_a_glove_index_matrix, related_h_seq_lengths, related_a_seq_lengths, related_labels, max_input_lengths, unrelated_labels

def report_pipeline_score(sub2_actual, sub2_preds, sub1_actual):
    '''
        sub2_actual: vectorized array with actual labels from subproblem 2
        sub2_preds: vectorized array with predicted labels from subproblem 2
        sub1_actual: vectorized array witha actual labels from the examples we took out for prob 2
    '''

    actual = sub2_actual + sub1_actual
    num_sub1 = len(sub1_actual)
    preds = sub2_preds + [3]*num_sub1
    report_score(actual, preds)

def main(debug=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--nn_weights', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--bimpmp', type=bool, default=False)
    args = parser.parse_args()
    weightFn = args.nn_weights
    output_fn = args.output_file
    assert weightFn
    isBimpmp = args.bimpmp
    if isBimpmp:
        config = BimpmpConfig()
    else:
        config = BiDirAttnBidirCondConfig()

    # Create result directories
    linear_related_preds = classify_related_unrelated()
    glove_matrix, related_h_glove_index_matrix, related_a_glove_index_matrix, related_h_seq_lengths, related_a_seq_lengths, max_input_lengths, related_labels, unrelated_labels = create_sub_class_test_data(linear_related_preds, config)
    test_set = [related_h_glove_index_matrix, related_a_glove_index_matrix, related_h_seq_lengths, related_a_seq_lengths, related_labels]
    sub1_labels = vectorize_stances(unrelated_labels)
    with tf.Graph().as_default():
        print 80 * "="
        print "INITIALIZING"
        print 80 * "="
        print "Building model...",
        scoring_fn = lambda actual, preds: report_pipeline_score(actual, preds, sub1_labels)
        if isBimpmp:
            model = Bimpmp(config, scoring_fn, max_input_lengths, glove_matrix, debug)
        else:
            model = Bidirectional_Attention_Conditonal_Encoding_LSTM_Model(config, scoring_fn, max_input_lengths, glove_matrix, debug)
        model.print_params()
        # Initialize variables
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            saver = create_tensorflow_saver(model.exclude_names)
            saver.restore(session, weightFn)
            # Finalize graph
            session.graph.finalize()
            print "TESTING"
            test_score, preds, test_confusion_matrix_str = model.predict(session, test_set, save_preds=True, UseShuffle=False)
            print preds
            print test_confusion_matrix_str
            with open(output_file, 'w') as file:
                file.write(test_confusion_matrix_str)
                file.write(preds)

if __name__ == "__main__":
    main(False)

