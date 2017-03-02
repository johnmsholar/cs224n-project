#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
CS 224N 2016-2017 
util.py: General utility routines.
Sahil Chopra <schopra8@cs.stanford.edu>
Saachi Jain <saachi@cs.stanford.edu>
John Sholar <jmsholar@cs.stanford.edu>
"""

import itertools
import random

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import os
from models import Labels, LABEL_MAPPING
from models.model import Model
from models.util import generate_batch, Progbar
from models.featurizer import read_binaries

class Config:
    """Holds model hyperparams and data information.
    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation. Use self.config.? instead of Config.?
    """
    # TODO: Change input_dim and num_classes appropriately (currently set for training example)
    initial_embedding_dim = 300
    reduced_embedding_dim = 100
    input_dim = 200
    num_classes = 4

    dropout = 0.5
    batch_size = 1000
    lr = .01
    n_epochs = 100

class SNLI_Baseline_NN(Model):
    """Baseline Neural Network described in the SNLI Corpus Paper (Bowman, et. al 2015).

    Specifically, the model takes in two 300-dim representations of the article
    headline and the article body itself. Then we project these into 100 dimensional
    space, concanenate these two vectors together, feed this 200-dim vector into a
    3 layer neural network (where each layer has a tanh activation function), and
    finally run the concanenated represnetation through a softmax.
    """

    def __init__(self, config):
        self.config = config

        # Defining placeholders.
        self.articles_placeholder = None
        self.headlines_placeholder = None
        self.labels_placeholder = None
        self.dropout_placeholder = None

        # Matrix Names
        self.weight_names = ['W1', 'W2', 'W3']
        self.bias_names = ['b1', 'b2', 'b3']

        self.build()

    def add_placeholders(self):
        """ Generates placeholder variables to represent the input tensors.
        """
        self.articles_placeholder = tf.placeholder(tf.float32, (None, self.config.initial_embedding_dim))
        self.headlines_placeholder = tf.placeholder(tf.float32, (None, self.config.initial_embedding_dim))
        self.labels_placeholder = tf.placeholder(tf.float32, (None, self.config.num_classes))
        self.dropout_placeholder = tf.placeholder(tf.float32)

    def create_feed_dict(self, articles_batch, headlines_batch, labels_batch=None, dropout=1):
        """Creates the feed_dict for the model.
        """
        if label_batch is not None:
            feed_dict = {
                self.articles_placeholder: articles_batch,
                self.headlines_placeholder: headlines_batch,
                self.dropout_placeholder: dropout,
                self.labels_placeholder: labels_batch,
            }

        else:
             feed_dict = {
                self.articles_placeholder: articles_batch,
                self.headlines_placeholder: headlines_batch,
                self.dropout_placeholder: dropout,
              }

        return feed_dict

    def add_prediction_op(self): 
        """Runs the SNLI Basline NN on the input.

        Returns:
            preds: tf.Tensor of shape (batch_size, 1)
        """

        # Declare initial tf.Variable matrices to map from initial embeddings to reduced dimension embeddings
        init_article_weights = tf.get_variable("init_article_weights", 
            shape=[self.config.initial_embedding_dim, self.config.reduced_embedding_dim],
            initializer=tf.contrib.layers.xavier_initializer()
        )

        init_article_bias = tf.get_variable("init_article_bias",
            shape=[self.config.reduced_embedding_dim],
            initializer=tf.constant_initializer(0)
        )

        init_headline_weights = tf.get_variable("init_headline_weights", 
            shape=[self.config.initial_embedding_dim, self.config.reduced_embedding_dim],
            initializer=tf.contrib.layers.xavier_initializer()
        )

        init_headline_bias = tf.get_variable("init_headline_bias",
            shape=[self.config.reduced_embedding_dim],
            initializer=tf.constant_initializer(0)
        )

        # Construct joint article + headline embedding
        init_article_layer = tf.nn.tanh(tf.matmul(articles_placeholder, init_article_weights) + init_article_bias)
        init_headline_layer = tf.nn.tanh(tf.matmul(headlines_placeholder, init_headline_weights) + init_headline_bias)
        embedding = tf.concat(values=[init_article_layer, init_headline_layer], concat_dim=1)

        # Declare tf.Variable weight matrices and bias vectors for 3 tanh layers
        weights = [
            tf.get_variable(n,
                shape=[self.config.inputs_batch, self.config.input_dim],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            for n in self.weight_names
        ]

        biases = [
            tf.get_variable(n,
                shape=[self.config.input_dim],
                initializer=tf.constant_initializer(0),
            )
            for n in self.bias_names
        ]

        # Declare tf.Variable weight matrix and bias vector for final layer
        final_weights = tf.get_variable("final_weights",
            shape=[self.config.input_dim, self.config.num_classes],
            initializer=tf.contrib.layers.xavier_initializer()
        )

        final_biases = tf.get_variable("final_biases",
            shape=[self.config.num_classes],
            initializer=tf.constant_initializer(0),
        )

        # Declare intermediate variables as products of base variables for tanh layers and final layer
        layer_1 = tf.nn.tanh(tf.matmul(embedding, weights[0]) + biases[0])
        layer_2 = tf.nn.tanh(tf.matmul(layer_1, weights[1]) + biases[1])
        layer_3 = tf.nn.tanh(tf.matmul(layer_2, weights[2]) + biases[2])
        final_layer = tf.matmul(layer_3, final_weights) + final_biases

    def add_loss_op(self, final_layer):
        """Adds Ops for the loss function to the computational graph.

        Args:
            pred: A tensor of shape (batch_size, n_classes) containing the output of the neural
                  network before the softmax layer.
        Returns:
            loss: A 0-d tensor (scalar)
        """
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=self.labels_placeholder,
                logits=final_layer)
        )
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

    def train_on_batch(self, sess, articles_batch, headlines_batch, labels_batch):
        feed = self.create_feed_dict(
            articles_batch,
            headlines_batch,
            labels_batch=labels_batch,
            dropout=self.config.dropout
        )

        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss


    # TODO EDIT
    def run_epoch(self, sess, train_examples, dev_set):
        prog = Progbar(target=1 + len(train_examples) / self.config.batch_size)
        for i, (train_x, train_y) in enumerate(minibatches(train_examples, self.config.batch_size)):
            loss = self.train_on_batch(sess, train_x, train_y)
            prog.update(i + 1, [("train loss", loss)])

        print "Evaluating on dev set",
        dev_UAS, _ = parser.parse(dev_set)
        print "- dev UAS: {:.2f}".format(dev_UAS * 100.0)
        return dev_UAS

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

def main(debug=True):
    print 80 * "="
    print "INITIALIZING"
    print 80 * "="
    config = Config()

    # Load Data
    parser, embeddings, train_examples, dev_set, test_set = load_and_preprocess_data(debug)
    
    if not os.path.exists('./data/weights/'):
        os.makedirs('./data/weights/')

    with tf.Graph().as_default():
        print "Building model...",
        start = time.time()
        print "took {:.2f} seconds\n".format(time.time() - start)

        init = tf.global_variables_initializer()
        saver = None if debug else tf.train.Saver()

        with tf.Session() as session:
            parser.session = session
            session.run(init)

            print 80 * "="
            print "TRAINING"
            print 80 * "="
            model.fit(session, saver, train_examples, dev_set)

            if not debug:
                print 80 * "="
                print "TESTING"
                print 80 * "="
                print "Restoring the best model weights found on the dev set"
                saver.restore(session, './data/weights/stance.weights')
                print "Final evaluation on test set",
                UAS, dependencies = parser.parse(test_set)
                print "- test UAS: {:.2f}".format(UAS * 100.0)
                print "Writing predictions"
                with open('q2_test.predicted.pkl', 'w') as f:
                    cPickle.dump(dependencies, f, -1)
                print "Done!"

if __name__ == '__main__':
    main(False)



"""
# Generate Sample Dataset in 2-dimensional space
# Class 1 is a Gaussian Centered Around (0, -5)
# Class 2 is a Gaussian Centered Around (0, 5)

centroid_1 = np.array([0, -1])
centroid_2 = np.array([0, 1])
cov = np.array([
  [1, 0],
  [0, 1]
])
size = 500
x1, y1 = np.random.multivariate_normal(centroid_1, cov, size).T
x2, y2 = np.random.multivariate_normal(centroid_2, cov, size).T
labels_1 = np.concatenate([np.array([[1, 0]]) for _ in range(size)], axis=0)
labels_2 = np.concatenate([np.array([[0, 1]]) for _ in range(size)], axis=0)
x = np.concatenate([x1, x2], axis = 0).reshape((-1, 1))
y = np.concatenate([y1, y2], axis = 0).reshape((-1, 1))
all_data = np.concatenate([x, y], axis=1)
all_labels = np.concatenate([labels_1, labels_2], axis=0)

# Split example data into train and test sets
train_data, test_data, train_labels, test_labels = train_test_split(
  all_data, all_labels, test_size=0.2, random_state=42
)
"""






# Run TF Session
# 1000 Train Iterations on Sample Data

sess = tf.Session()
sess.run(tf.global_variables_initializer())
X_train_input, X_test_input, y_train_input, y_test_input = read_binaries()
train_article_input_matrix, train_headlines_placeholder = X_train_input
test_article_input_matrix, test_headlines_placeholder = X_test_input

print('Binaries Initialized')
for iteration in range(n_epochs):
  article_batch, headline_batch, label_batch = generate_batch(
    train_article_input_matrix,
    train_headlines_placeholder,
    y_train_input
  )


  print('Iteration: ' + str(iteration))
  _ = sess.run(
    [train_step],
    feed_dict = {
      articles_placeholder : article_batch,
      headlines_placeholder : headline_batch,
      labels_placeholder : label_batch
    }
  )
  loss_value = sess.run(
    [loss],
    feed_dict = {
      articles_placeholder : train_article_input_matrix,
      headlines_placeholder : train_headlines_placeholder,
      labels_placeholder : y_train_input
    }
  )
  print('Loss: ' + str(loss_value))

# Evaluate performance on test set
print sess.run(
  loss, 
  feed_dict= { 
    articles_placeholder : test_article_input_matrix,
    headlines_placeholder : test_headlines_placeholder,
    labels_placeholder : y_test_input
  }
)

# Plot data
"""
plt.plot(x1, y1, 'x')
plt.plot(x2, y2, 'x')
plt.axis('equal')
plt.savefig('plot.png', bbox_inches='tight')
"""
