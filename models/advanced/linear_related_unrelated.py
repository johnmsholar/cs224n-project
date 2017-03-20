#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
CS 224N 2016-2017
linear_related_unrelated.py:
    A simple linear model to differentiate between related and unrelated
    (headline, article) pairs.
Sahil Chopra <schopra8@cs.stanford.edu>
Saachi Jain <saachi@cs.stanford.edu>
John Sholar <jmsholar@cs.stanford.edu>
"""

import sys
import argparse
import scipy
import os
import sklearn
import random
import numpy as np
import matplotlib.pyplot as plt


sys.path.insert(0, '../')

from fnc1_utils.generate_test_splits import compute_splits
from fnc1_utils import RELATED_UNRELATED_MAPPING, LABEL_MAPPING

from baselines.linear_baseline import generate_feature_files, generate_feature_matrices

from models.util import save_confusion_matrix

from models.fnc1_utils.score import report_score


def convert_to_two_class_problem(y_train, y_test, y_dev):
    inverted_label_mapping = dict(
        (value, key) for (key, value) in LABEL_MAPPING.items())
    two_class_name_mapping = {
        'agree': 'related',
        'disagree': 'related',
        'discuss': 'related',
        'unrelated': 'unrelated'
    }
    two_class_mapping = dict((key, two_class_name_mapping[value])
                             for (key, value) in inverted_label_mapping.items())
    for group in (y_train, y_test, y_dev):
        for index, label in enumerate(group):
            group[index] = RELATED_UNRELATED_MAPPING[two_class_mapping[label]]
    return y_train, y_test, y_dev

def evaluate_model(clf, X_train, X_test, X_dev, y_train, y_test,
                   y_dev, cm_output_prefix):
    # Compute and print confusion matrix
    y_train_predicted = clf.predict(X_train)
    y_test_predicted = clf.predict(X_test)
    y_dev_predicted = clf.predict(X_dev)
    cm_train = sklearn.metrics.confusion_matrix(y_train, y_train_predicted)
    cm_test = sklearn.metrics.confusion_matrix(y_test, y_test_predicted)
    cm_dev = sklearn.metrics.confusion_matrix(y_dev, y_dev_predicted)
    print('TRAIN CONFUSION MATRIX')
    print(cm_train)
    print('DEV CONFUSION MATRIX')
    print(cm_dev)
    print('TEST CONFUSION MATRIX')
    print(cm_test)
    classes = ['RELATED', 'UNRELATED']
    # plot_confusion_matrix(cm, classes, normalize=True)
    save_confusion_matrix(cm_train, classes,
                          cm_output_prefix + '_train.png', normalize=True)
    save_confusion_matrix(cm_test, classes,
                          cm_output_prefix + '_test.png', normalize=True)
    save_confusion_matrix(cm_dev, classes,
                          cm_output_prefix + '_dev.png', normalize=True)
    # Compute and print 5-Fold Cross Validation F1 Score
    weighted_f1 = sklearn.metrics.make_scorer(sklearn.metrics.f1_score, average='weighted')
    train_score = sklearn.model_selection.cross_val_score(clf, X_train, y_train,
                                                          scoring=weighted_f1,
                                                          cv=5)
    test_score = sklearn.model_selection.cross_val_score(clf, X_test, y_test,
                                                          scoring=weighted_f1,
                                                          cv=5)
    dev_score = sklearn.model_selection.cross_val_score(clf, X_dev, y_dev,
                                                         scoring=weighted_f1,
                                                         cv=5)

    print('TRAIN CROSS VALIDATION F1 SCORE')
    print(train_score)
    print('TRAIN FNC1 Official Score:')
    report_score(y_train, y_train_predicted)
    print('TEST CROSS VALIDATION F1 SCORE')
    print(test_score)
    print('TEST FNC1 Official Score:')
    report_score(y_test, y_test_predicted)
    print('DEV CROSS VALIDATION F1 SCORE')
    print(dev_score)
    print('DEV FNC1 Official Score:')
    report_score(y_dev, y_dev_predicted)

def train_model(X_train, y_train, model=None):
    if model == 'mnb':
        clf = sklearn.naive_bayes.MultinomialNB()
    elif model == 'svm':
        clf = sklearn.svm.SVC()
    elif model == 'randomforest':
        clf = sklearn.ensemble.RandomForestClassifier()
    else:
        clf = sklearn.naive_bayes.MultinomialNB()
    clf.fit(X_train, y_train)
    return clf

def main(args):
    if args.feature_output:
        generate_feature_files(args.feature_output, args, args.full)
    if args.feature_input and args.x_output and args.y_output:
        generate_feature_matrices(args.feature_input, args.x_output,
                                  args.y_output, args, args.full)
    if args.eval and args.x_input and args.y_input:
        (X_indices, y, b_id_to_article, h_id_to_headline,
         h_id_b_id_to_stance, raw_article_id_to_b_id,
         headline_to_h_id) = compute_splits()
        ((X_train_indices, X_test_indices, X_dev_indices),
         (y_train, y_test, y_dev)) = X_indices, y
        (y_train, y_test, y_dev) = convert_to_two_class_problem(
            y_train, y_test, y_dev)
        X_vectors = scipy.io.mmread(args.x_input).toarray()
        X_train, X_dev, X_test = create_feature_matrices(
            X_train_indices, X_test_indices, X_dev_indices,
            h_id_b_id_to_stance, X_vectors)
        if args.uniform_split:
            #y_train_matrix = dist_matrix(y_train)
            #y_test_matrix = dist_matrix(y_test)
            #y_dev_matrix = dist_matrix(y_dev)
            X, y = ((X_train, X_dev, X_test),
                    (y_train, y_dev, y_test))
            X, y = produce_uniform_data_split(X, y)
            ((X_train, X_dev, X_test),
             (y_train, y_dev, y_test)) = X, y
            #y_train = class_vector(y_train)
            #y_test = class_vector(y_test)
            #y_dev = class_vector(y_dev)
        clf = train_model(X_train, y_train, args.classifier)
        evaluate_model(clf, X_train, X_test, X_dev, y_train, y_test, y_dev,
                       args.cm_prefix)
    if args.plot_feature_dist:
        plot_feature_distribution(args)

def produce_uniform_data_split(X, y):
    X_train = X[0]
    X_dev = X[1]
    X_test = X[2]
    y_train = y[0]
    y_dev = y[1]
    y_test = y[2]
    target_variables = [
        (X_train, y_train),
        (X_dev, y_dev),
        (X_test, y_test)
    ]
    result_X = []
    result_y = []
    for X, y in target_variables:
        X = X.toarray()
        X_new = X
        y_new = y
        for row in range(X.shape[0]):
            if y[row] == 0:
                new_row = X[row, :].reshape(1, X_new.shape[1])
                X_new = np.concatenate([X_new, new_row], axis=0)
                X_new = np.concatenate([X_new, new_row], axis=0)
                y_new.append(0)
                y_new.append(0)
        result_X.append(X_new)
        result_y.append(y_new)
    X = tuple(result_X)
    y = tuple(result_y)
    return X, y

def create_feature_matrices(X_train_indices, X_test_indices, X_dev_indices,
                            h_id_b_id_to_stance, X_vectors, dev=True):
    vector_ordering = sorted(h_id_b_id_to_stance.keys())
    vector_index_mapping = dict((key, index) for index, key in enumerate(vector_ordering))
    X_train_matrix_indices = [vector_index_mapping[index] for index in X_train_indices]
    X_test_matrix_indices = [vector_index_mapping[index] for index in X_test_indices]
    X_dev_matrix_indices = [vector_index_mapping[index] for index in X_dev_indices]
    X_train = X_vectors[X_train_matrix_indices, :]
    X_test = X_vectors[X_test_matrix_indices, :]
    X_dev = X_vectors[X_dev_matrix_indices, :]
    if dev:
        return X_train, X_dev, X_test
    else:
        X_train = np.concatenate([X_train, X_dev], axis=0)
    return X_train, X_test

def plot_histogram(filename, data, bins, title, xlabel='Value', ylabel='Count'):
    plt.hist(data, bins=bins, histtype='bar', rwidth=.75)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_two_histograms(filename, data1, data2, label1, label2, bins, title,
                        xlabel='Value', ylabel='Count'):
    n, bins_1, patches = plt.hist(
        data2, bins=bins, histtype='bar', rwidth=.75, label=label2, alpha=.5)
    plt.hist(data1, bins=bins_1, histtype='bar', rwidth=.75, label=label1, alpha=.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(filename)
    plt.close()

def plot_feature_distribution(args):
    (X_indices, y, b_id_to_article, h_id_to_headline,
     h_id_b_id_to_stance, raw_article_id_to_b_id,
     headline_to_h_id) = compute_splits(random=False)
    ((X_train_indices, X_test_indices, X_dev_indices),
     (y_train, y_test, y_dev)) = X_indices, y
    (y_train, y_test, y_dev) = convert_to_two_class_problem(
        y_train, y_test, y_dev)
    X_vectors = scipy.io.mmread(args.x_input).toarray()
    X_train, X_dev, X_test = create_feature_matrices(
        X_train_indices, X_test_indices, X_dev_indices,
        h_id_b_id_to_stance, X_vectors)
    X = np.concatenate([X_train.toarray(), X_dev.toarray(), X_test.toarray()], axis=0)
    y = np.concatenate([np.array(y_train), np.array(y_dev), np.array(y_test)], axis=0)
    for i in range(X.shape[1]):
        filename= args.plot_prefix + '{0}.png'.format(i)
        feature_x = X[:, i]
        feature_x_0 = [x for index, x in enumerate(feature_x) if y[index] == 0]
        feature_x_1 = [x for index, x in enumerate(feature_x) if y[index] == 1]
        plot_two_histograms(filename, feature_x_0, feature_x_1, 'Related', 'Unrelated', 20, filename)
    x = 1

def classify_related_unrelated():
    (X_indices, y, b_id_to_article, h_id_to_headline,
     h_id_b_id_to_stance, raw_article_id_to_b_id,
     headline_to_h_id) = compute_splits(random=False)
    ((X_train_indices, X_dev_indices, X_test_indices),
     (y_train, y_dev, y_test)) = X_indices, y
    (y_train, y_test, y_dev) = convert_to_two_class_problem(
        y_train, y_test, y_dev)
    X_vectors = scipy.io.mmread(
        'data/linear_related_unrelated/matrices/tfidf_revised.mtx').toarray()
    X_train, X_test = create_feature_matrices(
        X_train_indices, X_test_indices, X_dev_indices,
        h_id_b_id_to_stance, X_vectors, dev=False)
    y_train = np.concatenate([y_train, y_dev], axis=0)
    clf = train_model(X_train, y_train, 'svm')
    y_test_predicted = clf.predict(X_test)
    y_test_final = np.logical_not(y_test_predicted.astype(bool))
    cm_test = sklearn.metrics.confusion_matrix(y_test, y_test_predicted)
    print(cm_test)
    return y_test_final


def parse_args():
    parser = argparse.ArgumentParser(description='Train and Test Linear Model')
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--uniform-split', action = 'store_true')
    parser.add_argument('--plot-feature-dist', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--classifier')
    parser.add_argument('--feature-output')
    parser.add_argument('--feature-input')
    parser.add_argument('--x-output')
    parser.add_argument('--y-output')
    parser.add_argument('--x-input')
    parser.add_argument('--y-input')
    parser.add_argument('--cm-prefix')
    parser.add_argument('--plot-prefix')
    feature_names = ['--overlap-features', '--overlap-features-clean',
                     '--bleu-score-features', '--bleu-score-features-clean',
                     '--tfidf-features', '--tfidf-features-clean',
                     '--headline-gram-features',
                     '--cross-gram-features', '--cross-gram-features-clean',
                     '--cross-gram-features-count']
    for name in feature_names:
        parser.add_argument(name, action = 'store_true')
    args = parser.parse_args()
    if not args.full:
        args.full = False
    return args

if __name__ == '__main__':

    #args = parse_args()
    #main(args)
    classify_related_unrelated()

