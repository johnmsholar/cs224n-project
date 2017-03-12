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

from advanced.attention_conditional_lstm import Attention_Conditonal_Encoding_LSTM_Model
from advanced.conditional_lstm import Conditonal_Encoding_LSTM_Model
from advanced.two_lstm_encoders import Two_LSTM_Encoders_Model
from advanced_model import create_data_sets_for_model, produce_uniform_data_split
from fnc1_utils.score import report_score
from fnc1_utils.featurizer import create_embeddings
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
        self.n_epochs = 5
        self.lr = 0.001
        self.max_grad_norm = 5.
        self.dropout_rate = 1.0
        self.beta = 0

def run_model(config, max_input_lengths, glove_matrix):
    """ Run the model.
    """
    with tf.Graph().as_default():
        print 80 * "="
        print "INITIALIZING"
        print 80 * "="

        # Create and configure model
        print "Building model...",
        start = time.time()
        model = Conditonal_Encoding_LSTM_Model(config, report_score, max_input_lengths, glove_matrix)
        model.print_params()
        print "took {:.2f} seconds\n".format(time.time() - start)

        # Initialize variables
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

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

            print 80 * "="
            print "TESTING"
            print 80 * "="
            print "Restoring the best model weights found on the dev set"
            saver.restore(session, model.best_weights_fn)

            print "Final evaluation on test set",
            test_score, _, test_confusion_matrix_str = model.predict(session, test_set, save_preds=True)
            with open(model.test_confusion_matrix_str, 'w') as file:
                file.write(test_confusion_matrix_str)
            print "- test Score: {:.2f}".format(test_score)

def main():
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--restore', action='store_true')
    args = parser.parse_args()

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

    train_examples, dev_set, test_set = create_data_sets_for_model(X, y)
    print "Distribution of Train {}".format(np.sum(train_examples[4], axis=0))
    print "Distribtion of Dev {}".format(np.sum(dev_set[4], axis=0))
    print "Distribution of Test{}".format(np.sum(test_set[4], axis=0))


    # Define hyperparameters
    hyperparameters = {
        'lr': [.1, .01, .001],
        'dropout_rate': [.5, .8, 1],
        'beta': [0, .5, 1, 10],
        'n_epochs': 5,
    }

    # Run model over all these hyper parameters
    for hyperparam in hyperparameters.keys():
        config = Config()
        for val in hyperparameters[hyperparam]:
            if hyperparam == 'lr':
                config.lr = val
            elif hyperparam == 'dropout_rate':
                config.dropout_rate = val
            elif hyperparam == 'beta':
                config.beta = val
            elif hyperparam == 'n_epochs':
                config.n_epochs = val
            run_model(config, max_input_lengths, glove_matrix)

if __name__ == '__main__':
    main(False)