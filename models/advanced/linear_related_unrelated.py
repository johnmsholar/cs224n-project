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

sys.path.insert(0, '../')

from fnc1_utils.generate_test_splits import compute_splits
from fnc1_utils import RELATED_UNRELATED_MAPPING, LABEL_MAPPING

from baselines.linear_baseline import generate_feature_vectors, retrieve_feature_vectors

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

def evaluate_model(clf, X_train, X_test, X_dev, y_train, y_test, y_dev):
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
                          'data/linear_related_unrelated/linear_related_unrelated_train_cm.png',
                          normalize=True)
    save_confusion_matrix(cm_test, classes,
                          'data/linear_related_unrelated/linear_related_unrelated_test_cm.png',
                          normalize=True)
    save_confusion_matrix(cm_dev, classes,
                          'data/linear_related_unrelated/linear_related_unrelated_dev_cm.png',
                          normalize=True)
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
    if args.x_output and args.y_output:
        generate_feature_vectors(args.x_output, args.y_output, args.full, args)
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
        clf = train_model(X_train, y_train)
        evaluate_model(clf, X_train, X_test, X_dev, y_train, y_test, y_dev)

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
    parser.add_argument('--x-output')
    parser.add_argument('--y-output')
    parser.add_argument('--x-input')
    parser.add_argument('--y-input')
    feature_names = ['--overlap-features', '--bleu-score-features',
                     '--tfidf-features', '--headline-gram-features',
                     '--cross-gram-features']
    for name in feature_names:
        parser.add_argument(name, action = 'store_true')
    args = parser.parse_args()
    if not args.full:
        args.full = False
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
