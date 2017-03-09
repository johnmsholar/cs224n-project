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

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import argparse
import cPickle
import time
import itertools
import random
import os
import sys
sys.path.insert(0, '../')

from fnc1_utils.score import report_score
from model import Model
from fnc1_utils.featurizer import create_inputs_by_glove
from util import Progbar, vectorize_stances, minibatches, create_tensorflow_saver

class Config:
    """Holds model hyperparams and data information.
    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation. Use self.config.? instead of Config.?
    """
    num_classes = 4 # Number of classses for classification task.
    embed_size = 300 # Size of Glove Vectors

    # Hyper Parameters
    headline_max_length = 1000 # Should be overriden in main
    article_max_length = 1000  # Should be overriden in main
    hidden_size = 400

    batch_size = 30
    n_epochs = 5
    lr = 0.02
    dropout_rate = 0.5

    # Other params
    pretrained_embeddings = None

class Two_LSTM_Encoders_Model(Model):
    """ 1 LSTM to encode headline.
        1 LSTM to encode article.
        Concatenate final hidden representations and feed through and MLP
        to determine final prediction.
    """

    def __init__(self, config):
        self.config = config

        # Defining placeholders.
        self.headline_seq_lengths = None
        self.article_seq_lengths = None

        self.headline_inputs_placeholder = None
        self.article_inputs_placeholder = None

        self.labels_placeholder = None
        self.dropout_placeholder = None

        # Construct single embedding matrix
        self.embedding_matrix = tf.constant(self.config.pretrained_embeddings, dtype=tf.float32, name="embedding_matrix")
        self.build()

        # Compute max over the prediction logits
        self.argmax_preds = tf.argmax(self.pred, axis=1)

    def add_placeholders(self):
        """ Generates placeholder variables to represent the input tensors.
        """
        self.headline_seq_lengths = tf.placeholder(tf.int32, (None), name="headline_seq_lengths")
        self.article_seq_lengths = tf.placeholder(tf.int32, (None), name="article_seq_lengths")

        self.headline_inputs_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.headline_max_length), name="headline_inputs_placeholder")
        self.article_inputs_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.article_max_length), name="article_inputs_placeholder")

        self.labels_placeholder = tf.placeholder(tf.float32, (None, self.config.num_classes), name="labels")
        self.dropout_placeholder = tf.placeholder(tf.float32, name="dropout")

    def create_feed_dict(self, headlines_batch, articles_batch, labels_batch=None, dropout=1, headline_seq_lengths=[], article_seq_lengths=[]):
        """Creates the feed_dict for the model.
        """
        feed_dict = {
            self.headline_inputs_placeholder: headlines_batch,
            self.article_inputs_placeholder: articles_batch,
            self.headline_seq_lengths: headline_seq_lengths,
            self.article_seq_lengths: article_seq_lengths,
            self.dropout_placeholder: dropout,
        }

        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch 

        return feed_dict

    def add_embedding(self, headline_model=True):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:

        Returns:
            embeddings: tf.Tensor of shape (None, max_length, embed_size)
        """
        if headline_model:
            e = tf.nn.embedding_lookup(self.embedding_matrix, self.headline_inputs_placeholder)
            embeddings = tf.reshape(e, shape=[-1, self.config.headline_max_length, self.config.embed_size], name="headline_embeddings")
        else:
            e = tf.nn.embedding_lookup(self.embedding_matrix, self.article_inputs_placeholder)
            embeddings = tf.reshape(e, shape=[-1, self.config.article_max_length, self.config.embed_size], name="article_embeddings")

        return embeddings

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
            headline_outputs, _ = tf.nn.dynamic_rnn(headline_cell, headlines_x, dtype=tf.float32, sequence_length = self.headline_seq_lengths)
            headline_output = headline_outputs[:, -1, :]
            assert headline_output.get_shape().as_list() == [None, self.config.hidden_size], "predictions are not of the right shape. Expected {}, got {}".format([None, self.config.hidden_size], headline_output.get_shape().as_list())

        # Articles -- Compute the output at the end of the LSTM (automatically unrolled)
        with tf.variable_scope("article_cell"):
            article_cell = tf.contrib.rnn.LSTMBlockCell(num_units=self.config.hidden_size)
            article_outputs, _ = tf.nn.dynamic_rnn(article_cell, articles_x, dtype=tf.float32, sequence_length = self.article_seq_lengths)
            article_output = article_outputs[:, -1, :]
            assert article_output.get_shape().as_list() == [None, self.config.hidden_size], "predictions are not of the right shape. Expected {}, got {}".format([None, self.config.hidden_size], article_output.get_shape().as_list())

        # Compute dropout on both headlines and articles
        headline_output_dropout = tf.nn.dropout(headline_output, dropout_rate)
        article_output_dropout = tf.nn.dropout(article_output, dropout_rate)

        # Concatenate headline and article outputs
        output = tf.concat(1, [headline_output_dropout, article_output_dropout])
        preds = tf.matmul(output, U) + b
        assert preds.get_shape().as_list() == [None, self.config.num_classes], "predictions are not of the right shape. Expected {}, got {}".format([None, self.config.num_classes], preds.get_shape().as_list())
        return preds

    def add_loss_op(self, preds):
        """Adds Ops for the loss function to the computational graph.

        Args:
            pred: A tensor of shape (batch_size, n_classes) containing the output of the neural
                  network before the softmax layer.
        Returns:
            loss: A 0-d tensor (scalar)
        """
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels_placeholder, logits=preds))
        return loss


    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train.

        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        optimizer = tf.train.AdamOptimizer(self.config.lr)
        train_op = optimizer.minimize(loss)
        return train_op    


    def train_on_batch(self, sess, headlines_batch, articles_batch, labels_batch):
        """Perform one step of gradient descent on the provided batch of data.

        Args:
            sess: tf.Session()
            headines_batch: list of ndarrays (1 per example)
                          each array consists of the glove-indices of the words
                          in the given headline
            articles_batch: list of ndarrays (1 per example)
                          each array consists of the glove-indices of the words
                          in the given article                          
        Returns:
            loss: loss over the batch (a scalar)
        """
        headline_seq_lengths = [len(input_arr) for input_arr in headlines_batch]
        article_seq_lengths = [len(input_arr) for input_arr in articles_batch]

        feed = self.create_feed_dict(
            headlines_batch,
            articles_batch,
            labels_batch=labels_batch,
            headline_seq_lengths=headline_seq_lengths,
            article_seq_lengths=article_seq_lengths,
            dropout=self.config.dropout_rate
        )
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def predict_on_batch(self, sess, headlines_batch, articles_batch):
        """Make predictions for the provided batch of data

        Args:
            sess: tf.Session()
        Returns:
            predictions: np.ndarray of shape (n_samples, n_classes)
        """
        headline_seq_lengths = [len(input_arr) for input_arr in headlines_batch]
        article_seq_lengths = [len(input_arr) for input_arr in articles_batch]
        feed = self.create_feed_dict(
            headlines_batch,
            articles_batch,
            headline_seq_lengths=headline_seq_lengths,
            article_seq_lengths=article_seq_lengths,
        )
        preds = sess.run(self.argmax_preds, feed_dict=feed)
        return preds

    # Note here that train_examples and dev_set are lists of three elements
    # headlines, articles, labels.
    def run_epoch(self, sess, train_examples, dev_set):
        prog = Progbar(target=1 + len(train_examples[0])/ self.config.batch_size)
        for i, (headlines_batch, articles_batch, labels_batch) in enumerate(minibatches(train_examples, self.config.batch_size)):
            loss = self.train_on_batch(sess, headlines_batch, articles_batch, labels_batch)
            prog.update(i + 1, [("train loss", loss)])

        print "Evaluating on dev set"
        prog = Progbar(target=1 + len(dev_set[0])/ self.config.batch_size)
        actual = vectorize_stances(dev_set[2])
        preds = []
        for i, (headlines_batch, articles_batch, labels_batch) in enumerate(minibatches(dev_set, self.config.batch_size)):
            predictions_batch = list(self.predict_on_batch(sess, headlines_batch, articles_batch))
            preds.extend(predictions_batch)
            prog.update(i + 1)     
        dev_score = report_score(actual, preds)
        print "- dev Score: {:.2f}".format(dev_score)
        return dev_score

    def fit(self, sess, saver, train_examples, dev_set):
        best_dev_score = 0
        for epoch in range(self.config.n_epochs):
            print "Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs)
            dev_score = self.run_epoch(sess, train_examples, dev_set)
            if dev_score > best_dev_score:
                best_dev_score = dev_score
                if saver:
                    print "New best dev! Saving model in ./data/weights/two_lstm_encoders_best_stance.weights"
                    saver.save(sess, './data/weights/two_lstm_encoders_best_stance.weights')
            if saver:
                print "Finished Epoch ... Saving model in ./data/weights/two_lstm_encoders_curr_stance.weights"
                saver.save(sess, './data/weights/two_lstm_encoders_curr_stance.weights')
        print

def main(debug=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--restore', action='store_true')
    args = parser.parse_args()

    if not os.path.exists('./data/weights/'):
        os.makedirs('./data/weights/')

    if not os.path.exists('./data/predictions/'):
        os.makedirs('./data/predictions/')

    with tf.Graph().as_default():
        print 80 * "="
        print "INITIALIZING"
        print 80 * "="
        config = Config()

        # Load Data
        # Note: X_train_input, X_dev_input, X_test_input are tuples of (articles, headlines).
        # where both articles and headlines are lists of example.
        # Each example is a sparse representation of the haddline or article, where the text 
        # is encoded as a series of indices into the glove-vectors.
        # y_train_input, y_dev_input, y_test_input are matrices (num_examples, num_classes)
        X_train_input, X_dev_input, X_test_input, y_train_input, y_dev_input, y_test_input, glove_matrix, max_lengths = create_inputs_by_glove(concatenate=False)
        config.headline_max_length = max_lengths[0]
        config.article_max_length = max_lengths[1]
        print "Headline Max Length is {}".format(config.headline_max_length)
        print "Article Max Length is {}".format(config.article_max_length)
        print "------------------------------------------"

        # Create Basic LSTM Model
        config.pretrained_embeddings = glove_matrix
        model = Two_LSTM_Encoders_Model(config)
        
        # Create Data Lists
        train_examples = [X_train_input[0], X_train_input[1], y_train_input]
        dev_set = [X_dev_input[0], X_dev_input[1], y_dev_input]
        test_set = [X_test_input[0], X_test_input[1], y_test_input]

        print "Building model...",
        start = time.time()
        print "took {:.2f} seconds\n".format(time.time() - start)

        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            session.run(init)
            exclude_names = set(["embedding_matrix:0", "embedding_matrix/Adam:0", "embedding_matrix/Adam_1:0"])
            saver = create_tensorflow_saver(exclude_names)
            if args.restore:
                saver.restore(session, './data/weights/two_lstm_encoders_curr_stance.weights')
                print "Restored weights from ./data/weights/two_lstm_encoders_curr_stance.weights"
                print "-------------------------------------------"
            session.graph.finalize()

            print 80 * "="
            print "TRAINING"
            print 80 * "="
            model.fit(session, saver, train_examples, dev_set)

            if saver:
                print 80 * "="
                print "TESTING"
                print 80 * "="
                print "Restoring the best model weights found on the dev set"
                saver.restore(session, './data/weights/two_lstm_encoders_best_stance.weights')
                print "Final evaluation on test set",

                actual = vectorize_stances(test_set[2])
                preds = list(model.predict_on_batch(session, *test_set[:2]))
                test_score = report_score(actual, preds)

                print "- test Score: {:.2f}".format(test_score)
                print "Writing predictions"
                with open('./data/predictions/two_lstm_encoders_predicted.pkl', 'w') as f:
                    cPickle.dump(preds, f, -1)
                print "Done!"

if __name__ == '__main__':
    main(False)
