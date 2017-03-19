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

class Config:
    """Holds model hyperparams and data information.
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
    beta = 0.02

    # Other params
    pretrained_embeddings = None

def load_data():
    # Load Data 
    # -- first data split into "unrelated" versus "related"
    # -- second data split amongst the "related" classes "agree", "disagree", "discuss"
    X_train_unrelated_split, X_dev_unrelated_split, X_test_unrelated_split, y_train_unrelated_split, y_dev_unrelated_split, y_test_unrelated_split, glove_matrix, max_input_lengths_unrelated_split, word_to_glove_index = create_inputs_by_glove_split_on_class(truncate=True, split_on_unrelated=True)
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

def main(debug=True):
    # Create result directories
    if not os.path.exists('./data/weights/'):
        os.makedirs('./data/weights/')

    if not os.path.exists('./data/predictions/'):
        os.makedirs('./data/predictions/')

    if not os.path.exists('./data/plots/'):
        os.makedirs('./data/plots/')

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--restore', action='store_true')
    args = parser.parse_args()

    # Load Data
    model_1_data, model_2_data, glove_matrix = load_data()

    # Create configs
    config_1 = Config()
    if args.epoch:
        config_1.n_epochs = args.epoch
        config_2.n_epochs = args.epoch

    config_1.pretrained_embeddings = glove_matrix
    config_1.h_max_length = model_1_data[3][0]
    config_1.b_max_length = model_1_data[3][1]

    config_2.pretrained_embeddings = glove_matrix
    config_2.h_max_length = model_2_data[3][0]
    config_2.b_max_length = model_2_data[3][1]

    with tf.Graph().as_default():
        print 80 * "="
        print "INITIALIZING"
        print 80 * "="
        
        print "Building model 1 ...",
        # TODO: Insert Model 1 Here
        model_1 = None
        start = time.time()
        print "took {:.2f} seconds\n".format(time.time() - start)

        init = tf.global_variables_initializer()
        saver = None if debug else tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            saver = create_tensorflow_saver(model_1.exclude_names)
            if args.restore:
                saver.restore(session, './data/weights/conditional_lstm_currstance.weights')
            session.graph.finalize()

            print 80 * "="
            print "TRAINING"
            print 80 * "="
            model_1.fit(session, saver, model_1_data[0], model_1_dev_examples[1])

    with tf.Graph().as_default():
        print 80 * "="
        print "INITIALIZING"
        print 80 * "="
        
        print "Building model 2 ...",
        # TODO: Insert Model 2 Here
        model_2 = None
        start = time.time()
        print "took {:.2f} seconds\n".format(time.time() - start)

        init = tf.global_variables_initializer()
        saver = None if debug else tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            saver = create_tensorflow_saver(model_2.exclude_names)
            if args.restore:
                saver.restore(session, './data/weights/conditional_lstm_curr_stance.weights')
            session.graph.finalize()

            print 80 * "="
            print "TRAINING"
            print 80 * "="
            model_1.fit(session, saver, model_2_data[0], model_2_dev_examples[1])
