#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
CS 224N 2016-2017 
featurizer.py: Constructs feature representations of articles and headlines.
Sahil Chopra <schopra8@cs.stanford.edu>
Saachi Jain <saachi@cs.stanford.edu>
John Sholar <jmsholar@cs.stanford.edu>
"""

from sklearn.model_selection import train_test_split
import numpy as np
from collections import defaultdict

from __init__ import LABEL_MAPPING
from enum import Enum
import csv
import re
import string

RANDOM_STATE = 42
TEST_SIZE = .20
GLOVE_SIZE = 300

SPACE_CHAR = ' '
NEWLINE_CHAR = '\n'
DASH_CHAR = '-'

# Raw training data from FNC-1 Challenge
TRAIN_BODIES_FNAME = "../fnc_1/train_bodies.csv"
TRAIN_STANCES_FNAME = "../fnc_1/train_stances.csv"

# Word Vector Representations
GLOVE_FILENAME = "../../glove.6B/glove.6B.300d.txt"

# Output embeddings and embedding references
BODY_EMBEDDING_FNAME = "../fnc_1/glove_body_matrix"
HEADLINE_EMBEDDING_FNAME = "../fnc_1/glove_headline_matrix"
ID_ID_STANCES_FNAME = "../fnc_1/id_id_stance.csv"

# Train/Test Split as defined by FNC-1 Baseline
TRAINING_FNAME = '../splits/training_ids.txt'
HOLD_OUT_FNAME = '../splits/hold_out_ids.txt'

# Train/Test Splits as defined by FNC-1 Baseline
# utilizing our indexing scheme.
TRAINING_SPLIT_FNAME = '../splits/custom_training_ids.txt'
TEST_SPLIT_FNAME = '../splits/custom_hold_out_ids.txt'

ID_ID_STANCES_TRAINING_SPLIT_FNAME = '../splits/id_id_stance_training.csv'
ID_ID_STANCES_TEST_SPLIT_FNAME = '../splits/id_id_stance_test.csv'

def construct_binaries():
    b_id_to_body, h_id_to_headline, h_id_b_id_to_stance = construct_data_set()
    glove_vectors = read_glove_set()
    save_glove_sums_matrix(b_id_to_body, h_id_to_headline, glove_vectors)
    write_id_id_stance(h_id_b_id_to_stance)

def read_binaries():
    glove_body_matrix, glove_headline_matrix = read_glove_sums()
    id_map = read_id_id_stance()
    X_train, X_test, y_train, y_test = test_train_split(id_map, TEST_SIZE)
    X_train_input = compute_id_embeddings(X_train, glove_body_matrix, glove_headline_matrix)
    X_test_input = compute_id_embeddings(X_train, glove_body_matrix, glove_headline_matrix)
    y_train_input = compute_stance_embeddings(y_train)
    y_test_input = compute_stance_embeddings(y_train)
    # returns tuples
    return X_train_input, X_test_input, y_train_input, y_test_input

def construct_data_set():
    # File Headers
    body_id_header = 'Body ID'
    article_body_header = 'articleBody'
    headline_header = 'Headline'
    stance_header = 'Stance'

    # Mappings
    b_id_to_index = {}
    b_id_to_body = {}
    h_id_to_headline = {}
    h_id_b_id_to_stance = {}

    # Read Article Bodies
    with open(TRAIN_BODIES_FNAME) as bodies_file:
        bodies_reader = csv.DictReader(bodies_file, delimiter = ',')
        index = 0
        for row in bodies_reader:
            b_id = int(row[body_id_header])
            b_id_to_index[b_id] = index
            article_body = row[article_body_header]
            article = clean(article_body)
            b_id_to_body[index] = article
            index+=1

    # Read Headline, ID -> Stance Mappings
    with open(TRAIN_STANCES_FNAME) as stances_file:
        stances_reader = csv.DictReader(stances_file, delimiter = ',')
        for h_id, row in enumerate(stances_reader):
            headline = row[headline_header]
            b_id = int(row[body_id_header])
            h_id_to_headline[h_id] = headline
            h_id_b_id_to_stance[(h_id, b_id_to_index[b_id])] = LABEL_MAPPING[row[stance_header]]

    return b_id_to_body, h_id_to_headline, h_id_b_id_to_stance
    # X_train, X_test, y_train, y_test = test_train_split(h_id_b_id_to_stance, TEST_SIZE)
    # return b_id_to_body, h_id_to_headline, h_id_b_id_to_stance, X_train, X_test, y_train, y_test

# Read the glove data set and return as a dict from word to numpy array
def read_glove_set():
    default = np.zeros(GLOVE_SIZE)
    glove_vectors = defaultdict(lambda: default)
    id_to_glove_body = {}
    with open(GLOVE_FILENAME) as glove_files:
        glove_reader = csv.reader(glove_files, delimiter = ' ', quotechar=None)
        index =0
        for row in glove_reader:
            word = row[0]
            # print (word, index)
            vec = np.array([float(i) for i in row[1:]])
            glove_vectors[word] = vec
            index+=1
            if index % 1000 == 0:
                print str(index/400000.0)
    return glove_vectors

# Read the dicts id to body and id to headline
# For each example, compute the sum of the glove vectors
# for each word. Create two numpy matrices to store
# the sums per example and save as binaries
def save_glove_sums_matrix(id_to_body, id_to_headline, glove_vectors):
    # read body
    glove_body_matrix = np.zeros((len(id_to_body), GLOVE_SIZE))
    for (body_id, text) in id_to_body.items():
        body_sum = np.zeros((GLOVE_SIZE))
        for word in text:
            body_sum += glove_vectors[word]
        glove_body_matrix[body_id] = body_sum
    # read headline
    glove_headline_matrix = np.zeros((len(id_to_headline),GLOVE_SIZE))
    for (headline_id, text) in id_to_headline.items():
        headline_sum = np.zeros((GLOVE_SIZE))
        for word in text:
            headline_sum += glove_vectors[word]
        glove_headline_matrix[headline_id] = headline_sum
    # saves as np binaries glove_body | glove_headline
    np.save(BODY_EMBEDDING_FNAME, glove_body_matrix)
    np.save(HEADLINE_EMBEDDING_FNAME, glove_headline_matrix)

# given a dict of (id, id) -> stance
def write_id_id_stance(id_id_stance):
    with open(ID_ID_STANCES_FNAME, 'w') as csvfile:
        idwriter = csv.writer(csvfile, delimiter=' ')
        for ((h_id, b_id), stance) in id_id_stance.items():
            row = [h_id, b_id, stance.value]
            idwriter.writerow(row)

def read_id_id_stance():
    id_map = {}
    with open(ID_ID_STANCES_FNAME) as csvfile:
            id_reader = csv.reader(csvfile, delimiter = ' ', quotechar='|')
            for row in id_reader:
                h_id = int(row[0])
                b_id = int(row[1])
                stance = int(row[2])
                id_map[(h_id, b_id)] = stance
    return id_map

# Load the glove body + headline binaries into numpy matrices
def read_glove_sums():
    glove_body_matrix = np.load(BODY_EMBEDDING_FNAME+".npy")
    glove_headline_matrix = np.load(HEADLINE_EMBEDDING_FNAME+".npy")
    return glove_body_matrix, glove_headline_matrix

# id_id_list is [(h_id, b_id)]
def compute_id_embeddings(id_id_list, glove_body_matrix, glove_headline_matrix):
    input_matrix_body = np.zeros((len(id_id_list), GLOVE_SIZE))
    input_matrix_headline = np.zeros((len(id_id_list), GLOVE_SIZE))
    index = 0
    for (h_id, b_id) in id_id_list:
        body_vec = glove_body_matrix[b_id]
        headline_vec = glove_headline_matrix[h_id]
        input_matrix_body[index] = body_vec
        input_matrix_headline[index] = headline_vec
        index += 1
    return (input_matrix_body, input_matrix_headline)

def compute_stance_embeddings(stance_list):
    labels_matrix = np.zeros((len(stance_list), len(LABEL_MAPPING)))
    for i in range(0,len(stance_list)):
        labels_matrix[i][stance_list[i]] = 1
    return labels_matrix

def clean(article_body):
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

def read_fnc1_training_split():
    training_fnc1_ids = []
    with open(TRAINING_FNAME, 'rb') as csvfile:
        training_reader = csv.reader(csvfile, delimiter = ' ', quotechar='|')
        for l in training_reader:
            training_fnc1_ids.append(int(l[0]))
    return training_fnc1_ids

def read_fnc1_hold_out_split():
    hold_out_fnc1_ids = []
    with open(HOLD_OUT_FNAME, 'rb') as csvfile:
        training_reader = csv.reader(csvfile, delimiter = ' ', quotechar='|')
        for l in training_reader:
            hold_out_fnc1_ids.append(int(l[0]))
    return hold_out_fnc1_ids

def compute_splits():
    id_id_stance = read_id_id_stance()
    training_fnc1_ids = read_fnc1_training_split()
    hold_out_fnc1_ids = read_fnc1_hold_out_split()

    print training_fnc1_ids
    print "----------------"
    print hold_out_fnc1_ids

def save_splits():
    pass

if __name__ == '__main__':
    # construct_binaries()
    compute_splits()
