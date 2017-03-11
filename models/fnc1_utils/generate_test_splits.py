#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
CS 224N 2016-2017 
generate_test_splits.py: Construct test splits.
Sahil Chopra <schopra8@cs.stanford.edu>
Saachi Jain <saachi@cs.stanford.edu>
John Sholar <jmsholar@cs.stanford.edu>
"""
from __init__ import LABEL_MAPPING, RELATED_UNRELATED_MAPPING, RELATED_CLASS_MAPPING
from collections import defaultdict
import filenames
import random
import string
import csv
import re
import os

# Constants
SPACE_CHAR = ' '
NEWLINE_CHAR = '\n'
DASH_CHAR = '-'

rgen = random.Random()
rgen.seed(1489123)

def compute_splits(training=0.8, random=True):
    """ Construct train, dev, test splits.
        training (percentage assigned for training data)
        random (whether to create a random test/train split or use FNC1 split)
    """
    # Load raw data and normalize according to our conventions
    b_id_to_article, h_id_to_headline, h_id_b_id_to_stance, raw_article_id_to_b_id, headline_to_h_id = construct_data_set()
    num_articles = len(b_id_to_article.keys())

    # Generate random or FNC1 test/train split
    if (random):
        training_ids, hold_out_ids = generate_random_hold_out_split(num_articles, training)
    else:
        training_ids, hold_out_ids = generate_original_holdouts(raw_article_id_to_b_id)

    # Shuffle article ids       
    rgen.shuffle(training_ids)
    train_ids = set(training_ids[:int(training * len(training_ids))])
    dev_ids = set(training_ids[int(training * len(training_ids)):])

    # Construct splits
    x_train = []
    y_train = []
    x_dev = []
    y_dev = []
    x_test = []
    y_test = []

    for (id_pair, stance) in h_id_b_id_to_stance.items():
        if id_pair[1] in train_ids:
            x_train.append(id_pair)
            y_train.append(stance)
        elif id_pair[1] in dev_ids:
            x_dev.append(id_pair)
            y_dev.append(stance)
        else:
            x_test.append(id_pair)
            y_test.append(stance)

    print "Train: {}, Dev: {}, Test: {}".format(len(x_train), len(x_dev), len(x_test))
    
    X = (x_train, x_dev, x_test)
    y = (y_train, y_dev, y_test)
    return X, y, b_id_to_article, h_id_to_headline, h_id_b_id_to_stance, raw_article_id_to_b_id, headline_to_h_id

def construct_data_set():
    """ Read in the headlines, articles, and (headline, raw_article_id, stance) tuples.
        Assign each headline an index. Assign each article an index. 
        Save (headline, raw_article_id, stance) tuples as (h_id, b_id, stance)
    """
    # Raw Data File Headers
    body_id_header = 'Body ID'
    article_body_header = 'articleBody'
    headline_header = 'Headline'
    stance_header = 'Stance'

    # Mappings
    raw_article_id_to_b_id = {}
    b_id_to_article = {}
    h_id_to_headline = {}
    headline_to_h_id = {}
    h_id_b_id_to_stance = {}

    # Read articles, construct our own set of article ids,
    # and mappings from our article ids to articles.
    with open(filenames.TRAIN_BODIES_FNAME) as bodies_file:
        bodies_reader = csv.DictReader(bodies_file, delimiter = ',')
        for b_id, row in enumerate(bodies_reader):
            raw_article_id = int(row[body_id_header])
            raw_article_id_to_b_id[raw_article_id] = b_id
            article_body = row[article_body_header]
            article = clean(article_body)
            b_id_to_article[b_id] = article

    # Ensure that all articles have been read
    assert len(b_id_to_article.keys()) == 1683

    # Read (headline, raw_article_id, stance) tuples.
    # First assign each headline a unique id.
    # Then construct (h_id, b_id, stance) tuples.
    headlines = set([])
    h_id = 0
    with open(filenames.TRAIN_STANCES_FNAME) as stances_file:
        stances_reader = csv.DictReader(stances_file, delimiter = ',')
        for row in stances_reader:
            headline = row[headline_header]

            if headline not in headlines:
                # New headline, so create an h_id
                headlines.add(headline)
                h_id_to_headline[h_id] = clean(headline)
                headline_to_h_id[headline] = h_id
                h_id += 1

            # Create (h_id, b_id, stance) tuple
            curr_h_id = headline_to_h_id[headline]
            raw_article_id = int(row[body_id_header])
            b_id = raw_article_id_to_b_id[raw_article_id]
            h_id_b_id_to_stance[(curr_h_id, b_id)] = LABEL_MAPPING[row[stance_header]]

    # Random spot check for correctness
    random_headline = "Missing American journalist reportedly beheaded by Islamic State "
    random_h_id = headline_to_h_id[random_headline]
    assert random_h_id == 383
    assert raw_article_id_to_b_id[195] == 125
    assert h_id_b_id_to_stance[(383, 125)] == LABEL_MAPPING["discuss"]

    return b_id_to_article, h_id_to_headline, h_id_b_id_to_stance, raw_article_id_to_b_id, headline_to_h_id

def clean(article_body):
    """ Clean the article body (string text) and return as a list of tokens.
    """
    article_body = article_body.replace(NEWLINE_CHAR, SPACE_CHAR)
    article_body = article_body.replace(DASH_CHAR, SPACE_CHAR)    

    def clean_word(word):
        w = word.lower()
        tokens = re.findall(r"[\w']+|[.,!?;]", w)
        return [t.strip() for t in tokens if (t.isalnum() or t in string.punctuation) and t.strip() != '']

    cleaned_article = []
    for w in str.split(article_body, SPACE_CHAR):
        c_word = clean_word(w)
        if c_word is not SPACE_CHAR:
            cleaned_article.extend(c_word)

    return cleaned_article

def generate_random_hold_out_split (num_articles, training = 0.8):
    """ Generate random article split
    """
    article_ids = [i for i in range(0, num_articles)] # article ids are consecutive
    rgen.shuffle(article_ids)  # and shuffle that list
    training_ids = article_ids[:int(training * len(article_ids))]
    hold_out_ids = article_ids[int(training * len(article_ids)):]
    return training_ids, hold_out_ids

def generate_original_holdouts(raw_article_id_to_b_id):
    """ Returns a list of article ids for training and for hold out from original.
    """
    # Generate article ids for training and test (holdout)
    training_fnc_ids = read_list_of_ids(filenames.ORIG_TRAIN_BODY_ID_FNAME)
    hold_out_fnc_ids = read_list_of_ids(filenames.ORIG_HOLDOUT_BODY_ID_FNAME)

    # Convert raw article ids to b_ids 
    training_fnc_ids = [raw_article_id_to_b_id[raw_article_id] for raw_article_id in training_fnc_ids]
    hold_out_fnc_ids = [raw_article_id_to_b_id[raw_article_id] for raw_article_id in hold_out_fnc_ids]
    return training_fnc_ids, hold_out_fnc_ids

def read_list_of_ids(filename):
    """ Given a file with an ID on every line, return the list of IDs.
    """
    id_list = []
    with open(filename, 'rb') as csvfile:
        id_reader = csv.reader(csvfile, delimiter = ' ', quotechar='|')
        id_list = [int(l[0]) for l in id_reader]
    return id_list

def split_by_unrelated_versus_related(id_pairs, stances):
    """ id pairs are h_id, b_id separate the samples into 
        unrelated versus related 
    """
    by_stance = distribute_by_stance(id_pairs, stances)
    
    # Construct a list of all (h_id, b_id) pairs that have
    # "related" labels i.e. have stances "agree", "disagree",
    # or "discuss"
    related_ids = []
    related_ids.extend(by_stance[LABEL_MAPPING["agree"]])
    related_ids.extend(by_stance[LABEL_MAPPING["disagree"]])
    related_ids.extend(by_stance[LABEL_MAPPING["discuss"]])

    # Construct a list of all "unrelated" stances
    unrelated_ids = by_stance[LABEL_MAPPING["unrelated"]]

    # Construct joint list of ids
    ids = related_ids + unrelated_ids

    # Construct stance labels
    related_labels = [RELATED_UNRELATED_MAPPING["related"]] * len(related_ids)
    unrelated_labels = [RELATED_UNRELATED_MAPPING["unrelated"]] * len(unrelated_ids)
    labels = related_labels + unrelated_labels

    return ids, labels

def split_by_related_class(id_pairs, stances):
    """ id pairs are h_id, b_id
        separate the samples by agree/disagree/discuss 
    """
    by_stance = distribute_by_stance(id_pairs, stances)

    # Split up (h_id, b_id) tules according to the type
    # of related class that the stance belongs to, i.e.
    # "agree", "disagree", "discuss"    
    agree = by_stance[LABEL_MAPPING["agree"]]
    disagree = by_stance[LABEL_MAPPING["disagree"]]
    discuss = by_stance[LABEL_MAPPING["discuss"]]

    # Construct a lit of all "related" ids
    ids = agree + disagree + discuss

    # Construct stance labels
    agree_labels = [RELATED_CLASS_MAPPING["agree"]] * len(agree)
    disagree_labels = [RELATED_CLASS_MAPPING["disagree"]] * len(disagree)
    discuss_labels = [RELATED_CLASS_MAPPING["discuss"]] * len(discuss)
    labels = agree_labels + disagree_labels + discuss_labels

    return ids, labels

def distribute_by_stance(id_pairs, stances):
    """ return a dict with {stance -> (h_id, b_id)}
    """
    by_stance = {i: [] for i in range(0, 4)}
    assert len(id_pairs) == len(stances)
    for i, id_pair in enumerate(id_pairs):
        stance = stances[i]
        by_stance[stance].append(id_pair)
    return by_stance

def underRepresent(id_pairs, stances, perc_unrelated):
    """ id_pairs are h_id, b_id
        cut the unrelated samples until there are < perc_unrelated samples in the whole set
    """
    by_stance = distribute_by_stance(id_pairs, stances)
    rgen.shuffle(by_stance[LABEL_MAPPING["unrelated"]])
    curr_num_unrelated = len(by_stance[LABEL_MAPPING["unrelated"]])
    num_total = len(id_pairs)
    needed_num_unrelated = int(perc_unrelated * num_total)
    if (needed_num_unrelated < curr_num_unrelated):
        by_stance[LABEL_MAPPING["unrelated"]] = by_stance[LABEL_MAPPING["unrelated"]][:needed_num_unrelated]
    # (id_pair, stance)
    new_num_unrelated = len(by_stance[LABEL_MAPPING["unrelated"]])
    new_id_pairs_stance = []
    for (stance, stance_id_pair_list) in by_stance.items():
        new_id_pairs_stance += [(id_pair, stance) for id_pair in stance_id_pair_list]
    rgen.shuffle(new_id_pairs_stance)
    id_pairs = [sample[0] for sample in new_id_pairs_stance]
    stances = [sample[1] for sample in new_id_pairs_stance]
    print "Under-representing, so now {} samples from {}".format(new_num_unrelated, curr_num_unrelated)
    return id_pairs, stances