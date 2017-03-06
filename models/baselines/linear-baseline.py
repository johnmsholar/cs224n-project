#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
CS 224N 2016-2017 
linear-baseline.py: Linear baseline described in
                    SNLI Corpus paper (Bowman et. al 2015).
Sahil Chopra <schopra8@cs.stanford.edu>
Saachi Jain <saachi@cs.stanford.edu>
John Sholar <jmsholar@cs.stanford.edu>
"""
import sys
import argparse
import sklearn.naive_bayes
import sklearn.model_selection
import sklearn
import nltk
import itertools
import numpy as np
import scipy
import random

sys.path.insert(0, '../../')
from models.util import plot_confusion_matrix, save_confusion_matrix
from models.fnc1_utils.featurizer import construct_data_set
from models.fnc1_utils.score import report_score

rgen = random.Random()
rgen.seed(1489215)


# Compute train-test-dev split in a sanitary way
def compute_splits(id_id_stance, training=0.8):
    training_ids, hold_out_ids = generate_random_hold_out_split(id_id_stance, training)

    rgen.shuffle(training_ids)
    train_ids = training_ids[:int(training * len(training_ids))]
    dev_ids = training_ids[int(training * len(training_ids)):]


    train_ids = set(train_ids)
    dev_ids = set(dev_ids)
    # train pairs are (headline, body)
    x_train = []
    y_train = []
    x_dev = []
    y_dev = []
    x_test = []
    y_test = []
    for (id_pair, stance) in id_id_stance.items():
        if id_pair[1] in train_ids:
            x_train.append(id_pair)
            y_train.append(stance.value)
        elif id_pair[1] in dev_ids:
            x_dev.append(id_pair)
            y_dev.append(stance.value)
        else:
            x_test.append(id_pair)
            y_test.append(stance.value)
    return x_train, x_dev, x_test, y_train, y_dev, y_test


# generate random article split
# pass in the number of articles
def generate_random_hold_out_split (id_id_stance, training = 0.8):
    unique_article_ids = set([ids[1] for ids in id_id_stance])
    num_articles = len(unique_article_ids)
    article_ids = list(unique_article_ids)
    rgen.shuffle(article_ids)  # and shuffle that list
    training_ids = article_ids[:int(training * num_articles)]
    hold_out_ids = article_ids[int(training * num_articles):]
    return training_ids, hold_out_ids


# Generate modified BLEU scores for each (healdine, article) pair, in which BLEU
# score is evaluated for a series of overlapping segments of the article.
# See description below for more information.
def generate_bleu_score_features(b_id_to_body, h_id_to_headline, h_id_b_id_to_stance):

    # Slice the body text into overlapping segments of length BLEU_SEGMENT_LENGTH,
    #   such that each segment overlaps with half of the next segment (to ensure that
    #   no phrase shorter than BLEU_SEGMENT_LENGTH is scliced in two
    # For every segment, compute the BLEU score of the headline with respect to the segment,
    #   and return the maximum such score
    # Ideally, this allows us to examine if at any point, the short headline matches a
    #   segment of the much longer reference text, while avoiding the brevity penalty
    #   associated with comparing a short hypothesis to a long reference text.
    def max_segment_bleu_score(head, body):
        BLEU_SEGMENT_LENGTH = 2 * len(head)
        SEGMENT_INCREMENT = len(head)
        max_bleu_score = 0.0
        smoothing_function = nltk.translate.bleu_score.SmoothingFunction()
        for start_index in range(0, len(body), SEGMENT_INCREMENT):
            body_segment = body[start_index:start_index + BLEU_SEGMENT_LENGTH]
            score = nltk.translate.bleu_score.sentence_bleu(
                [body_segment], head, smoothing_function=smoothing_function.method1,
                weights=(0.333, 0.333, 0.333))
            max_bleu_score = max(score, max_bleu_score)
        return max_bleu_score

    def map_ids_to_feature_vector(ids):
        h_id, b_id = ids
        headline = h_id_to_headline[h_id]
        body = b_id_to_body[b_id]
        score = max_segment_bleu_score(headline, body)
        return {BLEU_FEATURE_NAME: score}

    num_pairs = len(h_id_b_id_to_stance.keys())
    bleu_score_feature_vectors = {}
    BLEU_FEATURE_NAME = 'max_segment_bleu_score'
    """
    # Attempts to parallelize this process gave no efficiency boost
    f = lambda k : (k, map_ids_to_feature_vector(k))
    bleu_score_feature_vectors = dict(map(f, h_id_b_id_to_stance.keys()))
    """
    for index, ids in enumerate(h_id_b_id_to_stance):
        if index % 100 == 0:
            print(float(index) / num_pairs)
        bleu_score_feature_vectors[ids] = map_ids_to_feature_vector(ids)
    return bleu_score_feature_vectors

# Not yet implemented
def generate_overlap_features(b_id_to_body, h_id_to_headline, h_id_b_id_to_stance):
    pass


# Generates indicator features for all unigrams and bigrams in the headline
def generate_headline_gram_features(b_id_to_body, h_id_to_headline, h_id_b_id_to_stance):
    HEADLINE_GRAM_FEATURE_NAME = 'headline_gram'
    headline_gram_features = {}
    num_pairs = len(h_id_b_id_to_stance)
    for index, (h_id, b_id) in enumerate(h_id_b_id_to_stance):
        if index % 100 == 0:
            print(float(index) / num_pairs)
        headline = h_id_to_headline[h_id]
        headline_unigrams = nltk.ngrams(headline, 1)
        headline_bigrams = nltk.ngrams(headline, 2)
        pair_headline_gram_features = {}
        for gram in itertools.chain(headline_unigrams, headline_bigrams):
            pair_headline_gram_features[(HEADLINE_GRAM_FEATURE_NAME, gram)] = 1.0
        headline_gram_features[(h_id, b_id)] = pair_headline_gram_features
    return headline_gram_features

# For every (headline, article) pair, Generate a feature vector containing indicator features for:
    #   1. Cross-Unigrams: every pair of words across the headline and article which share a POS tag
    #   2. Cross-Bigrams: every pair of bigrams across the healdine and article which share a POS tag on the second word
def generate_cross_gram_features(b_id_to_body, h_id_to_headline, h_id_b_id_to_stance):

    # For a single (headline, article) pair, generate a single feature vector composed of all cross-ngrams
    # matching the conditions described above
    def single_pair_cross_ngram_features(headline, article, n):
        CROSS_NGRAM_FEATURE_NAME = 'cross_ngram'
        headline_pos = nltk.pos_tag(headline)
        article_pos = nltk.pos_tag(article)
        all_pos = headline_pos + article_pos
        unique_pos_classes = set([token_pos[1] for token_pos in all_pos])
        result = {}
        for pos_class in unique_pos_classes:
            headline_matches = [g for i, g in enumerate(nltk.ngrams(headline, n)) if headline_pos[i + n - 1][1] == pos_class]
            article_matches = [g for i, g in enumerate(nltk.ngrams(article, n)) if article_pos[i + n - 1][1] == pos_class]
            for cross_gram in itertools.product(headline_matches, article_matches):
                result[(CROSS_NGRAM_FEATURE_NAME, cross_gram)] = 1.0
        return result

    def map_ids_to_feature_vector(ids):
        h_id, b_id = ids
        headline = h_id_to_headline[h_id]
        body = b_id_to_body[b_id]
        unigram_features = single_pair_cross_ngram_features(headline, body, 1)
        bigram_features = single_pair_cross_ngram_features(headline, body, 2)
        gram_features = dict(unigram_features.items() + bigram_features.items())
        return gram_features

    all_cross_gram_features = {}
    """
    # Attempts to parallelize this process gave no efficiency boost
    f = lambda k: (k, map_ids_to_feature_vector(k))
    all_cross_gram_features = dict(map(f, h_id_b_id_to_stance.keys()))
    """
    num_pairs = len(h_id_b_id_to_stance)
    for index, (h_id, b_id) in enumerate(h_id_b_id_to_stance):
        if index % 100 == 0:
            print(float(index) / num_pairs)
        all_cross_gram_features[(h_id, b_id)] = map_ids_to_feature_vector((h_id, b_id))
    return all_cross_gram_features

def join_features_on_key(feature_maps, h_id_b_id_to_stance, h_id_b_id_keys):
    all_keys_aggregated_features_dict = []
    for (h_id, b_id) in h_id_b_id_keys:
        key = (h_id, b_id)
        key_feature_vectors = [mapping[key].items() for mapping in feature_maps]
        aggregated_feature_vector = reduce(lambda x, y: x + y, key_feature_vectors)
        aggregated_features_dict = dict(aggregated_feature_vector)
        all_keys_aggregated_features_dict.append(aggregated_features_dict)
    return all_keys_aggregated_features_dict

def generate_feature_vectors(feature_matrix_filename, output_class_filename, full=False):
    b_id_to_body, h_id_to_headline, h_id_b_id_to_stance_superset = construct_data_set()
    h_id_to_headline = dict([(k, v.split()) for k, v in h_id_to_headline.items()])
    if not full:
        h_id_b_id_to_stance = dict(h_id_b_id_to_stance_superset.items()[:200])
    else:
        h_id_b_id_to_stance = h_id_b_id_to_stance_superset
    h_id_b_id_keys = h_id_b_id_to_stance.keys()
    print('DATASET CONSTRUCTED')

    bleu_score_features = generate_bleu_score_features(b_id_to_body, h_id_to_headline, h_id_b_id_to_stance)
    print('BLEU SCORE FEATURE VECTORS GENERATED')
    headline_gram_features = generate_headline_gram_features(b_id_to_body, h_id_to_headline, h_id_b_id_to_stance)
    print('HEADLINE GRAM FEATURE VECTORS GENERATED')
    cross_gram_features = generate_cross_gram_features(b_id_to_body, h_id_to_headline, h_id_b_id_to_stance)
    print('CROSS GRAM FEATURE VECTORS GENERATED')
    print('INDIVIDUAL FEATURE VECTORS GENERATED')

    feature_maps = [bleu_score_features, headline_gram_features, cross_gram_features]
    all_keys_aggregated_features_dict = join_features_on_key(feature_maps, h_id_b_id_to_stance, h_id_b_id_keys)
    print('GLOBAL FEATURE VECTORS GENERATED')

    v = sklearn.feature_extraction.DictVectorizer(sparse=True)
    X = v.fit_transform(all_keys_aggregated_features_dict)
    Y = np.array([(key, h_id_b_id_to_stance[key]) for key in h_id_b_id_keys])
    #Y = np.array(h_id_b_id_keys.items())
    scipy.io.mmwrite(feature_matrix_filename, X)
    np.save(output_class_filename, Y)
    print('FEATURE MATRIX SAVED TO {0}'.format(feature_matrix_filename))
    print('OUTPUT CLASS MATRIX SAVED TO {0}'.format(output_class_filename))


def train_model(X_train, y_train, model=None):
    if model == 'mnb':
        clf = sklearn.naive_bayes.MultinomialNB()
    else:
        clf = sklearn.naive_bayes.MultinomialNB()
    clf.fit(X_train, y_train)
    return clf


def retrieve_feature_vectors(feature_matrix_filename, output_class_filename):
    X = scipy.io.mmread(feature_matrix_filename).tocsr()
    h_id_b_id_stance = np.load(output_class_filename)
    h_id_b_id_stance_dict = dict(h_id_b_id_stance)
    ids_to_indices = dict([(id_id_stance[0], index) for index, id_id_stance in enumerate(h_id_b_id_stance)])
    h_id_b_id_keys = [elem[0] for elem in h_id_b_id_stance]
    y = np.array([elem[1] for elem in h_id_b_id_stance])
    x_train_ids, x_dev_ids, x_test_ids, y_train_list, y_dev_list, y_test_list = compute_splits(h_id_b_id_stance_dict)

    x_train_indices = [ids_to_indices[ids] for ids in x_train_ids]
    x_dev_indices = [ids_to_indices[ids] for ids in x_dev_ids]
    x_test_indices = [ids_to_indices[ids] for ids in x_test_ids]
    #y_train_indices = [ids_to_indices[ids] for ids in y_train_ids]
    #y_dev_indices = [ids_to_indices[ids] for ids in y_dev_ids]
    #y_test_indices = [ids_to_indices[ids] for ids in y_test_ids]

    X_train = X[x_train_indices, :]
    X_dev = X[x_dev_indices, :]
    X_test = X[x_test_indices, :]
    y_train = y[x_train_indices]
    y_dev = y[x_dev_indices]
    y_test = y[x_test_indices]

    return X_train, X_dev, X_test, y_train, y_dev, y_test


def evaluate_model(clf, X_train, X_test, y_train, y_test):
    # Compute and print confusion matrix
    y_predicted = clf.predict(X_test)
    cm = sklearn.metrics.confusion_matrix(y_test, y_predicted)
    print('CONFUSION MATRIX')
    print(cm)
    classes = ['AGREE', 'DISAGREE', 'DISCUSS', 'UNRELATED']
    # plot_confusion_matrix(cm, classes, normalize=True)
    save_confusion_matrix(cm, classes, 'cm.png', normalize=True)
    # Compute and print 5-Fold Cross Validation F1 Score
    weighted_f1 = sklearn.metrics.make_scorer(sklearn.metrics.f1_score, average='weighted')
    score = sklearn.model_selection.cross_val_score(clf, X_train, y_train, scoring=weighted_f1, cv=5)
    print('CROSS VALIDATION F1 SCORE')
    print(score)
    print('FNC1 Official Score:')
    report_score(y_test, y_predicted)


def parse_args():
    parser = argparse.ArgumentParser(description='Train and Test Linear Model')
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--x-output')
    parser.add_argument('--y-output')
    parser.add_argument('--x-input')
    parser.add_argument('--y-input')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if not (args.x_output is None or args.y_output is None):
        generate_feature_vectors(args.x_output, args.y_output, full=args.full)
    if not (args.x_input is None or args.y_input is None):
        X_train, X_dev, X_test, y_train, y_dev, y_test = retrieve_feature_vectors(args.x_input, args.y_input)
        clf = train_model(X_train, y_train)
        evaluate_model(clf, X_train, X_test, y_train, y_test)
