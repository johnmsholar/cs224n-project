#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
CS 224N 2016-2017 
advanced_model.py: 
    Model class that abstracts Tensorflow graph for learning tasks.
    This differes from model.py in that it specifically handles the cases where
    we want to handle headlines and articles as completely separate entities.
Sahil Chopra <schopra8@cs.stanford.edu>
Saachi Jain <saachi@cs.stanford.edu>
John Sholar <jmsholar@cs.stanford.edu>
"""

import numpy as np
import tensorflow as tf
from util import Progbar, minibatches, vectorize_stances, create_tensorflow_saver
from fnc1_utils.score import report_score

import os
import sys
import cPickle

class Advanced_Model(object):
    """ This model handles the majority case, where we want to process 
        the headlines and articles seperately.
    """

    def __init__(self, config, scoring_function, max_lengths, glove_matrix):
    	""" config must be Config class
        scoring_function(actual, preds)
    	"""
        # Init config
        self.config = config

        # Placeholders
        self.h_seq_lengths_placeholder = None
        self.a_seq_lengths_placeholder = None
        self.h_placeholder = None
        self.a_placeholder = None
        self.labels_placeholder = None
        self.dropout_placeholder = None

        # Constants
        self.embedding_matrix = tf.constant(
            glove_matrix,
            dtype=tf.float32,
            name="embedding_matrix"
        )
        
        # Save Files
        fn_names = self.get_fn_names()
        self.weights_path = './data/weights'
        self.preds_path = './data/predictions'
        self.plots_path = './data/plots'
        self.best_weights_fn = '{}/{}'.format(self.weights_path, fn_names[0])
        self.curr_weights_fn = '{}/{}'.format(self.weights_path, fn_names[1])
        self.preds_fn = '{}/{}'.format(self.preds_path, fn_names[2])

        # Create Necessary Directories 
        if not os.path.exists('./data/weights/'):
            os.makedirs('./data/weights/')

        if not os.path.exists('./data/predictions/'):
            os.makedirs('./data/predictions/')

        if not os.path.exists('./data/plots/'):
            os.makedirs('./data/plots/')

        # TODO: Potentially Unnecessary -- maybe remove this??
        self.exclude_names = set(
            [
            "embeddings_matrix:0",
            "embeddings_matrix_h:0",
            "embeddings_matrix_b:0"
            ]
        )

        # Scoring Function for Evaluation
        self.scoring_function = scoring_function

        # Configure internal params
        self.h_max_length = max_lengths[0]
        self.a_max_length = max_lengths[1]        

        # Build Tensorflow Graph Model
        self.build()

    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)
        self.class_predictions = tf.argmax(self.pred, axis=1)

    def get_fn_names(self):
        """ Retrieve file names.
            fn_names = [best_weights_fn, curr_weights_fn, preds_fn]
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_placeholders(self):
        """Adds placeholder variables to tensorflow computational graph.
        """
        self.h_seq_lengths_placeholder = tf.placeholder(
            tf.int32,
            (None),
            name="headline_seq_lengths"
        )
        self.a_seq_lengths_placeholder = tf.placeholder(
            tf.int32,
            (None),
            name="article_seq_lengths"
        )
        self.h_placeholder = tf.placeholder(
            tf.int32,
            shape=(None, self.h_max_length),
            name="headline_inputs_placeholder"
        )
        self.a_placeholder = tf.placeholder(
            tf.int32,
            shape=(None, self.a_max_length), 
            name="article_inputs_placeholder"
        )
        self.labels_placeholder = tf.placeholder(
            tf.float32,
            (None, self.config.num_classes),
            name="labels"
        )
        self.dropout_placeholder = tf.placeholder(
            tf.float32,
            name="dropout"
        )

    def create_feed_dict(
        self,
        headlines_batch,
        articles_batch,
        labels_batch=None,
        dropout=1,
        h_seq_lengths=[],
        a_seq_lengths=[]
    ):
        """Creates the feed_dict for the model.
        """
        feed_dict = {
            self.h_placeholder: headlines_batch,
            self.a_placeholder: articles_batch,
            self.h_seq_lengths_placeholder: h_seq_lengths,
            self.a_seq_lengths_placeholder: a_seq_lengths,
            self.dropout_placeholder: dropout,
        }
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch 
        return feed_dict

    def add_embedding(self, headline_embedding=True):
        """ Adds an embedding layer that maps from input tokens (integers) to vectors and then 
            concatenates those vectors.
        """
        if headline_embedding:
            e = tf.nn.embedding_lookup(self.embedding_matrix, self.h_placeholder)
            embeddings = tf.reshape(
                e,
                shape=[-1, self.h_max_length, self.config.embed_size],
                name="headline_embeddings"
            )
        else:
            e = tf.nn.embedding_lookup(self.embedding_matrix, self.a_placeholder)
            embeddings = tf.reshape(
                e,
                shape=[-1, self.a_max_length, self.config.embed_size],
                name="article_embeddings"
            )
        return embeddings

    def add_prediction_op(self):
        """Implements the core of the model that transforms a batch of input data into predictions.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_loss_op(self, preds):
        """Adds Ops for the loss function to the computational graph.
        """
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=self.labels_placeholder,
                logits=preds
            )
        )
        reg = 0
        for var in tf.trainable_variables():
            reg += tf.reduce_mean(tf.nn.l2_loss(var))
        reg *= self.config.beta
        loss += reg
        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.
        """
        optimizer = tf.train.AdamOptimizer(self.config.lr)
        train_op = optimizer.minimize(loss)
        return train_op

    def train_on_batch(self, sess, headlines_batch, articles_batch, labels_batch):
        """Perform one step of gradient descent on the provided batch of data.
        """
        h_seq_lengths = [len(arr) for arr in headlines_batch]
        a_seq_lengths = [len(arr) for arr in articles_batch]
        dropout_rate = self.config.dropout_rate
        feed = self.create_feed_dict(
            headlines_batch,
            articles_batch,
            labels_batch,
            dropout_rate,
            h_seq_lengths,
            a_seq_lengths
        )
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def predict_on_batch(self, sess, headlines_batch, articles_batch):
        """Make predictions for the provided batch of data.
        """
        h_seq_lengths = [len(arr) for arr in headlines_batch]
        a_seq_lengths = [len(arr) for arr in articles_batch]
        feed = self.create_feed_dict(
            headlines_batch,
            articles_batch,
            h_seq_lengths=h_seq_lengths,
            a_seq_lengths=a_seq_lengths
        )
        preds = sess.run(self.class_predictions, feed_dict=feed)
        return preds

    def predict(self, sess, data_set, save_preds=False):
        """ Compute predictions on a given data set.
            data_set = [headlines, articles, labels]
            Return predictions and score
        """
        # Compute Predictions
        prog = Progbar(target=1+len(data_set[0])/self.config.batch_size)
        actual = vectorize_stances(data_set[2])
        preds = []
        for i, (headlines_batch, articles_batch, _) in enumerate(minibatches(data_set, self.config.batch_size)):
            predictions_batch = list(self.predict_on_batch(sess, headlines_batch, articles_batch))
            preds.extend(predictions_batch)
            prog.update(i + 1)

        if save_preds:
            print "Writing predictions to {}".format(self.preds_fn)
            with open(self.preds_fn, 'w') as f:
                cPickle.dump(preds, f, -1)

        # Compute Score
        score = self.scoring_function(actual, preds)
        return score, preds   

    def run_epoch(self, sess, train_examples, dev_set):
        """ Run a single epoch on the given train_examples. 
            Then evaluate on the dev_set.
            Both train_examples and dev_set should be of format:
            [headlines, articles, labels]
        """
        # Train the Epoch
        prog = Progbar(target=1+len(train_examples[0])/self.config.batch_size)
        for i, (headlines_batch, articles_batch, labels_batch) in enumerate(minibatches(train_examples, self.config.batch_size)):
            loss = self.train_on_batch(sess, headlines_batch, articles_batch, labels_batch)
            prog.update(i + 1, [("train loss", loss)])

        # Evaluate on the Dev Set
        print "Evaluating on dev set"
        dev_score, _ = self.predict(sess, dev_set)
        print "- dev Score: {:.2f}".format(dev_score)
        return dev_score

    def fit(self, sess, saver, train_examples, dev_set):
        """ Train the model over self.config.n_epochs.
        """
        best_dev_score = 0
        for epoch in range(self.config.n_epochs):
            print "Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs)
            dev_score = self.run_epoch(sess, train_examples, dev_set)
            if dev_score > best_dev_score:
                best_dev_score = dev_score
                if saver:
                    print "New best dev! Saving model in {}".format(self.best_weights_fn)
                    saver.save(sess, self.best_weights_fn)
            if saver:
                print "Finished Epoch ... Saving model in {}".format(self.curr_weights_fn)
                saver.save(sess, self.curr_weights_fn)
            print    

def create_data_sets_for_model(
    X_train,
    X_dev,
    X_test,
    y_train,
    y_dev,
    y_test,
):
    """ Given train, dev, and test splits for input and labels,
        construct the arrays that can be processed by the model.
    """
    train_examples = [X_train[0], X_train[1], y_train]
    dev_set = [X_dev[0], X_dev[1], y_dev]
    test_set = [X_test[0], X_test[1], y_test]
    return train_examples, dev_set, test_set
