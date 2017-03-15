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
import sklearn
import random
import numpy as np

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
    if args.x_input and args.y_input:
        (X_indices, y, b_id_to_article, h_id_to_headline,
         h_id_b_id_to_stance, raw_article_id_to_b_id,
         headline_to_h_id) = compute_splits()
        ((X_train_indices, X_test_indices, X_dev_indices),
         (y_train, y_test, y_dev)) = X_indices, y
        (y_train, y_test, y_dev) = convert_to_two_class_problem(
            y_train, y_test, y_dev)
        X_vectors = scipy.io.mmread(args.x_input).tocsr()
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
        clf = train_model(X_train, y_train, args)
        evaluate_model(clf, X_train, X_test, X_dev, y_train, y_test, y_dev,
                       args.cm_prefix)

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

"""
def produce_uniform_data_split(X, y):
    X_train = X[0]
    X_dev = X[1]
    X_test = X[2]
    y_train = y[0]
    y_dev = y[1]
    y_test = y[2]

    def get_dist(y_vector):
        zero_count = len(filter(lambda v: v == 0, y))
        one_count = len(filter(lambda v: v == 1, y))
        return [zero_count, one_count]

    train_dist = get_dist(y_train)
    dev_dist = get_dist(y_dev)
    test_dist = get_dist(y_test)
    train_count = min(train_dist)
    dev_count = min(dev_dist)
    test_count = min(test_dist)

    target_variables = [
        (X_train, y_train, train_count),
        (X_dev, y_dev, dev_count),
        (X_test, y_test, test_count)
    ]
    finalized_variables = []
    for X, y, count in target_variables:
        dist = get_dist(y)
        new_X, new_y = None, None
        for i in range(2):
            rows_in_class = [index for index in range(y.shape[0]) if y[index] == i]

            rows_in_class = (y[:, i] == 1)
            num_rows_in_class = np.sum(rows_in_class.astype(int))

            X_h_seq_lengths = [l for i, l in enumerate(X[2]) if rows_in_class[i]]
            X_a_seq_lengths = [l for i, l in enumerate(X[3]) if rows_in_class[i]]
            X_local = (X[0][rows_in_class, :], X[1][rows_in_class, :], X_h_seq_lengths, X_a_seq_lengths)
            y_local = y[rows_in_class, :]

            random_indices = random.sample(range(num_rows_in_class), int(count))
            X_local_h_seq_lengths = [l for i, l in enumerate(X_local[2]) if i in random_indices]
            X_local_a_seq_lengths = [l for i, l in enumerate(X_local[3]) if i in random_indices]
            X_local = (X_local[0][random_indices, :], X_local[1][random_indices, :], X_local_h_seq_lengths, X_local_a_seq_lengths)
            y_local = y_local[random_indices, :]

            if new_X is None and new_y is None:
                new_X = X_local
                new_y = y_local
            else:
                new_X = (np.concatenate([new_X[0], X_local[0]]), np.concatenate([new_X[1], X_local[1]]), new_X[2] + X_local[2], new_X[3] + X_local[3])
                new_y = np.concatenate([new_y, y_local])
        finalized_variables.append((new_X, new_y))
    (X_train, y_train), (X_dev, y_dev), (X_test, y_test) = finalized_variables
    # need to shuffle
"""

def create_feature_matrices(X_train_indices, X_test_indices, X_dev_indices,
                            h_id_b_id_to_stance, X_vectors):
    vector_ordering = sorted(h_id_b_id_to_stance.keys())
    vector_index_mapping = dict((key, index) for index, key in enumerate(vector_ordering))
    X_train_matrix_indices = [vector_index_mapping[index] for index in X_train_indices]
    X_test_matrix_indices = [vector_index_mapping[index] for index in X_test_indices]
    X_dev_matrix_indices = [vector_index_mapping[index] for index in X_dev_indices]
    X_train = X_vectors[X_train_matrix_indices, :]
    X_test = X_vectors[X_test_matrix_indices, :]
    X_dev = X_vectors[X_dev_matrix_indices, :]
    return X_train, X_dev, X_test

def parse_args():
    parser = argparse.ArgumentParser(description='Train and Test Linear Model')
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--uniform-split', action = 'store_true')
    parser.add_argument('--classifier')
    parser.add_argument('--feature-output')
    parser.add_argument('--feature-input')
    parser.add_argument('--x-output')
    parser.add_argument('--y-output')
    parser.add_argument('--x-input')
    parser.add_argument('--y-input')
    parser.add_argument('--cm-prefix')
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
    args = parse_args()
    main(args)
