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
sys.path.insert(0, '../')

from fnc1_utils.featurizer import create_inputs_by_glove_split_on_class

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

def load_data():
    # Load Data 
    # -- first data split into "unrelated" versus "related"
    # -- second data split amongst the "related" classes "agree", "disagree", "discuss"
    _, X_dev_unrelated_split, X_test_unrelated_split, y_train_unrelated_split, y_dev_unrelated_split, y_test_unrelated_split, glove_matrix, max_input_lengths_unrelated_split, word_to_glove_index = create_inputs_by_glove_split_on_class(truncate=True, split_on_unrelated=True)
    X_train_related_split, X_dev_related_split, X_test_related_split, y_train_related_split, y_dev_related_split, y_test_related_split, _, max_input_lengths_related_split, _ = create_inputs_by_glove_split_on_class(truncate=True, split_on_unrelated=True, (word_to_glove_index, glove_matrix))

    model_1_train_examples = [X_train_unrelated_split[0], X_train_unrelated_split[1], y_train_unrelated_split]
    model_1_dev_examples = [X_dev_unrelated_split[0], X_dev_unrelated_split[1], y_dev_unrelated_split]
    model_1_test_examples = [X_test_unrelated_split[0], X_test_unrelated_split[1], y_test_unrelated_split]

    model_2_train_examples = [X_train_related_split[0], X_train_related_split[1], y_train_related_split]
    model_2_dev_examples = [X_dev_related_split[0], X_dev_related_split[1], y_dev_related_split]
    model_2_test_examples = [X_test_related_split[0], X_test_related_split[1], y_test_related_split]

    model_1_data = (model_1_train_examples, model_1_dev_examples, model_1_test_examples, max_input_lengths_unrelated_split)
    model_2_data = (model_2_train_examples, model_2_dev_examples, model_2_test_examples, max_input_lengths_related_split)

    return model_1_data, model_2_data, glove_matrix

def create_class1_data():
    # original hold_out
    X, y, b_id_to_article, h_id_to_headline, h_id_b_id_to_stance, raw_article_id_to_b_id, headline_to_h_id = compute_splits(random=False)
    X_test = X[2] # [h_id, b_id]
    Y_test = y[2] # stances



# def create_data_sets_for_model(X, y, debug=False):
#     """ Given train, dev, and test splits for input, sequnce lengths, labels,
#         construct the arrays that can be processed by the model.
#         X: [X_train, X_dev, X_test] X_train, X_dev, X_test are tuples consisting of (headline matrix of glove indices, article matrix of glove indices, h_seq_lengths, article_seq_lengths)
#         y: [y_train, y_dev, y_test] y_train, y_dev, y_test are matrices where each row is a 1 hot vector represntation of the class label

#         Note: Replicate examples if in debug mode.

#         Returns lists in the form of [headline_glove_index_matrix, article_glove_index_matrix, h_seq_lengths, a_seq_lengths, labels]
#     """
#     if debug:
#         train_examples = [np.repeat(X[0][0], 400, axis=0), np.repeat(X[0][1], 400, axis=0), X[0][2]*400, X[0][3]*400, np.repeat(y[0], 400, axis=0)]
#     else:    
#         train_examples = [X[0][0], X[0][1], X[0][2], X[0][3], y[0]]

#     dev_set = [X[1][0], X[1][1], X[1][2], X[1][3], y[1]]
#     test_set = [X[2][0], X[2][1], X[2][2], X[2][3], y[2]]
#     return train_examples, dev_set, test_set


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
    parse.add_argument('--nn_weights', type=str)
    parse.add_argument('--output_file', type=str)
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
    model = Bidirectional_Attention_Conditonal_Encoding_LSTM_Model
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

