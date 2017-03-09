#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
CS 224N 2016-2017 
featurizer.py: Constructs feature representations of articles and headlines.
Sahil Chopra <schopra8@cs.stanford.edu>
Saachi Jain <saachi@cs.stanford.edu>
John Sholar <jmsholar@cs.stanford.edu>
"""

from __init__ import LABEL_MAPPING
from enum import Enum
import csv
import re
import string
import filenames

import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict

from generate_test_splits import compute_splits, underRepresent

import sys
sys.path.insert(0, '../')
from util import Progbar

RANDOM_STATE = 42
TRAINING_SIZE = .80
GLOVE_SIZE = 300
MAX_BODY_LENGTH = 500
NUM_GLOVE_WORDS = 400000.0

NUM_GLOVE_WORDS_42B = 1.9e7

SPACE_CHAR = ' '
NEWLINE_CHAR = '\n'
DASH_CHAR = '-'
UNK_TOKEN = "PLACEHOLDER_UNK"
USE_RANDOM_FNC = False
UNDER_REPRESENT = True
PERC_UNRELATED = 0.5

# if concatenate is true then X's are one input matrix which have article and headline concatenated
# otherwise return tuple of input matrices for each input x
def create_inputs_by_glove(concatenate=True, truncate=True):
    b_id_to_body, h_id_to_headline, h_id_b_id_to_stance = construct_data_set()
    # X is [(headline id, body id)]
    X_train, X_dev, X_test, y_train, y_dev, y_test = compute_splits(h_id_b_id_to_stance, TRAINING_SIZE, USE_RANDOM_FNC)
    if (UNDER_REPRESENT):
        X_train, y_train = underRepresent(X_train, y_train, PERC_UNRELATED)
    # read glove
    glove_vectors = read_glove_set() # word to numpy array
    glove_vectors[UNK_TOKEN] = np.random.normal(size=GLOVE_SIZE)
    glove_words = glove_vectors.keys()
    word_to_glove_index = {} # mapping into the glove index for embedding
    glove_matrix = np.zeros((len(glove_vectors), GLOVE_SIZE))
    for i, word in enumerate(glove_words):
        word_to_glove_index[word] = i
        glove_matrix[i] = glove_vectors[word]

    # compute glove index vector for every headline
    # sample id -> computed glove indices
    h_id_to_glove_index_vector = compute_glove_index_vector(h_id_to_headline, word_to_glove_index, truncate)
    b_id_to_glove_index_vector = compute_glove_index_vector(b_id_to_body, word_to_glove_index, truncate)
    max_body_length = max([len(index_vec) for (b_id, index_vec) in b_id_to_glove_index_vector.items()])
    max_headline_length = max([len(index_vec) for (h_id, index_vec) in h_id_to_glove_index_vector.items()])
    max_input_lengths = (max_headline_length, max_body_length)

    X_train_input = compute_glove_id_embeddings(X_train, h_id_to_glove_index_vector, b_id_to_glove_index_vector, max_input_lengths, concatenate)
    X_dev_input = compute_glove_id_embeddings(X_dev, h_id_to_glove_index_vector, b_id_to_glove_index_vector, max_input_lengths, concatenate)
    X_test_input = compute_glove_id_embeddings(X_test, h_id_to_glove_index_vector, b_id_to_glove_index_vector, max_input_lengths, concatenate)


    y_train_input = compute_stance_embeddings(y_train)
    y_dev_input = compute_stance_embeddings(y_dev)
    y_test_input = compute_stance_embeddings(y_test)

    return X_train_input, X_dev_input, X_test_input, y_train_input, y_dev_input, y_test_input, glove_matrix, max_input_lengths


# id_to_text should be {id -> text} for either headline or body
# word_to_glove_index should be {word -> word's index in glove}
def compute_glove_index_vector (id_to_text, word_to_glove_index, truncate = True):
    sample_id_to_glove_index_vector = {}
    for (sample_id, text) in id_to_text.items():
        if truncate:
            trunc_text = text[:MAX_BODY_LENGTH]
        else:
            trunc_text = text
        index_vector = np.zeros(len(trunc_text))
        for i, word in enumerate(trunc_text):
            if word not in word_to_glove_index:
                word = UNK_TOKEN
            index_vector[i] = word_to_glove_index[word]
        sample_id_to_glove_index_vector[sample_id] = index_vector
    return sample_id_to_glove_index_vector

# return a matrix/es where each row is an element
# @max_input_lengths is headline length, body length
# if concanatenate is true:
#   for each sample concatenate the index vectors of the headline and the body (headline + body)
# otherwise return a tuple (headline_input_matrix, body_input_matrix)

def compute_glove_id_embeddings (id_list, h_id_to_glove_index_vector, b_id_to_glove_index_vector, max_input_lengths, concatenate):
    if concatenate:
        max_input_length = sum(max_input_lengths)
        input_list = np.zeros((len(id_list), max_input_length))
        for index, (h_id, b_id) in enumerate(id_list):
            h_index_vector = h_id_to_glove_index_vector[h_id]
            b_index_vector = b_id_to_glove_index_vector[b_id]
            headline_body_vec = np.concatenate((h_index_vector, b_index_vector))
            input_list[index][:len(headline_body_vec)] = headline_body_vec
        return input_list
    else:
        headline_list = np.zeros((len(id_list), max_input_lengths[0]))
        body_list = np.zeros((len(id_list), max_input_lengths[1]))
        for index, (h_id, b_id) in enumerate(id_list):
            h_index_vector = h_id_to_glove_index_vector[h_id]
            b_index_vector = b_id_to_glove_index_vector[b_id]
            headline_list[index][:len(h_index_vector)] = h_index_vector
            body_list[index][:len(b_index_vector)] = b_index_vector
        return (headline_list, body_list)

# construct binaries where each text is represented as the sum of the glove vectors for the words
def construct_glove_sum_binaries():
    b_id_to_body, h_id_to_headline, h_id_b_id_to_stance = construct_data_set()
    glove_vectors = read_glove_set()
    save_glove_sums_matrix(b_id_to_body, h_id_to_headline, glove_vectors)
    write_id_id_stance(h_id_b_id_to_stance)

# read the binaries where each text is represented as the sum of the glove vectors for the words
def read_glove_sum_binaries():
    glove_body_matrix, glove_headline_matrix = read_glove_sums()
    id_map = read_id_id_stance()
    X_train, X_dev, X_test, y_train, y_dev, y_test = compute_splits(id_map, TRAINING_SIZE, USE_RANDOM_FNC)
    X_train_input = compute_id_embeddings(X_train, glove_body_matrix, glove_headline_matrix)
    X_dev_input = compute_id_embeddings(X_dev, glove_body_matrix, glove_headline_matrix)
    X_test_input = compute_id_embeddings(X_test, glove_body_matrix, glove_headline_matrix)
    y_train_input = compute_stance_embeddings(y_train)
    y_dev_input = compute_stance_embeddings(y_dev)
    y_test_input = compute_stance_embeddings(y_test)
    return X_train_input, X_dev_input, X_test_input, y_train_input, y_dev_input, y_test_input

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
    with open(filenames.TRAIN_BODIES_FNAME) as bodies_file:
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
    with open(filenames.TRAIN_STANCES_FNAME) as stances_file:
        stances_reader = csv.DictReader(stances_file, delimiter = ',')
        for h_id, row in enumerate(stances_reader):
            headline = row[headline_header]
            b_id = int(row[body_id_header])
            h_id_to_headline[h_id] = headline
            h_id_b_id_to_stance[(h_id, b_id_to_index[b_id])] = LABEL_MAPPING[row[stance_header]]

    return b_id_to_body, h_id_to_headline, h_id_b_id_to_stance

# Read the glove data set and return as a dict from word to numpy array
def read_glove_set():
    print "STARTED READING GLOVE VECTORS INTO MEMORY"
    default = np.zeros(GLOVE_SIZE)
    glove_vectors = defaultdict(lambda: default)
    id_to_glove_body = {}
    with open(filenames.GLOVE_FILENAME) as glove_files:
        glove_reader = csv.reader(glove_files, delimiter = ' ', quotechar=None)
        prog = Progbar(target=NUM_GLOVE_WORDS)
        for index, row in enumerate(glove_reader):
            word = row[0]
            vec = np.array([float(i) for i in row[1:]])
            glove_vectors[word] = vec
            if index % 10000 == 0:
                prog.update(index)
    print ""
    print "FINISHED READING GLOVE VECTORS INTO MEMORY"
    print "------------------------------------------"
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
    np.save(filenames.BODY_EMBEDDING_FNAME, glove_body_matrix)
    np.save(filenames.HEADLINE_EMBEDDING_FNAME, glove_headline_matrix)

# given a dict of (id, id) -> stance
def write_id_id_stance(id_id_stance):
    with open(filenames.ID_ID_STANCES_FNAME, 'w') as csvfile:
        idwriter = csv.writer(csvfile, delimiter=' ')
        for ((h_id, b_id), stance) in id_id_stance.items():
            row = [h_id, b_id, stance]
            idwriter.writerow(row)

def read_id_id_stance():
    id_map = {}
    with open(filenames.ID_ID_STANCES_FNAME) as csvfile:
            id_reader = csv.reader(csvfile, delimiter = ' ', quotechar='|')
            for row in id_reader:
                h_id = int(row[0])
                b_id = int(row[1])
                stance = int(row[2])
                id_map[(h_id, b_id)] = stance
    return id_map

# Load the glove body + headline binaries into numpy matrices
def read_glove_sums():
    glove_body_matrix = np.load(filenames.BODY_EMBEDDING_FNAME+".npy")
    glove_headline_matrix = np.load(filenames.HEADLINE_EMBEDDING_FNAME+".npy")
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


if __name__ == '__main__':
    create_inputs_by_glove(concatenate=False)
    # read_binaries()
    # construct_glove_sum_binaries()
    # create_inputs_by_glove()
