#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
CS 224N 2016-2017 
model.py: Model class that abstracts Tensorflow graph for learning tasks.
Sahil Chopra <schopra8@cs.stanford.edu>
Saachi Jain <saachi@cs.stanford.edu>
John Sholar <jmsholar@cs.stanford.edu>
"""

import numpy as np
import tensorflow as tf
from util import minibatches, Progbar, vectorize_stances
from fnc1_utils.score import report_score

class Model(object):
    """Abstracts a Tensorflow graph for a learning task.

    We use various Model classes as usual abstractions to encapsulate tensorflow
    computational graphs.
    """
    def add_placeholders(self):
        """Adds placeholder variables to tensorflow computational graph.

        Tensorflow uses placeholder variables to represent locations in a
        computational graph where data is inserted.  These placeholders are used as
        inputs by the rest of the model building and will be fed data during
        training.

        See for more information:
        https://www.tensorflow.org/versions/r0.7/api_docs/python/io_ops.html#placeholders
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def create_feed_dict(self, articles_batch, headlines_batch, labels_batch=None):
        """Creates the feed_dict for one step of training.

        A feed_dict takes the form of:
        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        If labels_batch is None, then no labels are added to feed_dict.

        Args:
            articles_batch: A batch of article bodies.
            headlines_batch: A batch of headlines.
            labels_batch: A batch of label data.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_prediction_op(self):
        """Implements the core of the model that transforms a batch of input data into predictions.

        Returns:
            pred: A tensor of shape (batch_size, n_classes)
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_loss_op(self, final_layer):
        """Adds Ops for the loss function to the computational graph.

        Args:
            pred: A tensor of shape (batch_size, n_classes)
        Returns:
            loss: A 0-d tensor (scalar) output
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        sess.run() to train the model. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Args:
            loss: Loss tensor (a scalar).
        Returns:
            train_op: The Op for training.
        """

        raise NotImplementedError("Each Model must re-implement this method.")

    def train_on_batch(self, sess, articles_batch, headlines_batch, labels_batch):
        """Perform one step of gradient descent on the provided batch of data.

        Args:
            sess: tf.Session()
            articles_batch: np.ndarray of shape (n_samples, n_dim)
            headlines_batch: np.ndaray of shape (n_samples, n_dim)
            labels_batch: np.ndarray of shape (n_samples, n_classes)
        Returns:
            loss: loss over the batch (a scalar)
        """
        feed = self.create_feed_dict(articles_batch, headlines_batch, labels_batch=labels_batch)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def predict_on_batch(self, sess, articles_batch, headlines_batch):
        """Make predictions for the provided batch of data

        Args:
            sess: tf.Session()
            articles_batch: np.ndarray of shape (n_samples, n_dim)
            headlines_batch: np.ndaray of shape (n_samples, n_dim)
        Returns:
            predictions: np.ndarray of shape (n_samples, n_classes)
        """
        feed = self.create_feed_dict(articles_batch, headlines_batch)
        predictions = sess.run(self.pred, feed_dict=feed)
        preds = tf.argmax(predictions, axis=1).eval()
        return preds

    def run_epoch(self, sess, train_examples, dev_set):
        prog = Progbar(target=1 + train_examples[0].shape[0] / self.config.batch_size)
        for i, (articles_batch, headlines_batch, labels_batch) in enumerate(minibatches(train_examples, self.config.batch_size)):
            loss = self.train_on_batch(sess, articles_batch, headlines_batch, labels_batch)
            prog.update(i + 1, [("train loss", loss)])
        print "Evaluating on train set"
        train_actual = vectorize_stances(train_examples[2])
        train_preds = list(self.predict_on_batch(sess, *train_examples[:2]))
        train_score = report_score(train_actual, train_preds)
        print "Evaluating on dev set"
        actual = vectorize_stances(dev_set[2])
        preds = list(self.predict_on_batch(sess, *dev_set[:2]))
        dev_score = report_score(actual, preds)

        print "- train Score {:.2f}".format(train_score)
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
                    print "New best dev! Saving model in ./data/weights/stance.weights"
                    saver.save(sess, './data/weights/stance.weights')
            print


    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)
