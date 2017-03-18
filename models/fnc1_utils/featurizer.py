#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
CS 224N 2016-2017 
featurizer.py: Constructs feature representations of articles and headlines.
Sahil Chopra <schopra8@cs.stanford.edu>
Saachi Jain <saachi@cs.stanford.edu>
John Sholar <jmsholar@cs.stanford.edu>
"""

from __init__ import LABEL_MAPPING, RELATED_UNRELATED_MAPPING, RELATED_CLASS_MAPPING
from generate_test_splits import compute_splits, underRepresent, split_by_unrelated_versus_related, split_by_related_class

from collections import defaultdict
import filenames
import numpy as np
import csv
import sys

sys.path.insert(0, '../')
from util import Progbar

# Constants
NUM_GLOVE_WORDS = 400000.0
UNK_TOKEN = "PLACEHOLDER_UNK"

# Legacy -- Global Variables
TRAINING_SIZE = .80
GLOVE_SIZE = 100
#GLOVE_SIZE = 300
# GLOVE_SIZE = 2
MAX_BODY_LENGTH = 500
USE_RANDOM_FNC = False
UNDER_REPRESENT = False
PERC_UNRELATED = 0.5

def create_embeddings(
    training_size=.80,
    random_split=False,
    truncate_headlines=False,
    truncate_articles=True,
    classification_problem=1,
    max_headline_length=500,
    max_article_length=500,
    glove_set=None,
    debug_printing=False,
    debug=False
): 
    """
    training_size: train/test split config
    random_split: False -> use FNC1 split, True -> use random split
    truncate_headline: False -> full headline, True -> truncate headline
    truncate_articles: True -> truncate article, False -> full article
    classification_problem: 
                            1 -- 4 class prediction problem
                            2 -- "related" versus "unrelated" problem
                            3 -- "agree", "disagree", "discuss" problem
    max_headline_length: length at which we truncate headline, if we are truncating
    max_article_length: lenght at which we truncate article, if we are truncating
    glove_set: None -> Read Glove, Else (word_to_glove_index, glove_matrix) -> Don't read Glove

    Returns X, y, glove_matrix, max_input_lengths, word_to_glove_index
    X: [X_train, X_dev, X_test] X_train, X_dev, X_test are tuples consisting of (headline matrix of glove indices, article matrix of glove indices, h_seq_lengths, article_seq_lengths)
    y: [y_train, y_dev, y_test] y_train, y_dev, y_test are matrices where each row is a 1 hot vector represntation of the class label
    glove_matrix: embeddings matrix of words -> glove vectors
    word_to_glove:index: mapping from word to index in glove_matrix (row number)
    """
    if debug:
        global GLOVE_SIZE
        GLOVE_SIZE = 2

    # X is [X_train, X_dev, X_test] where each is of the format [(headline id, body id)]
    # y is [y_train, y_dev, y_test] where each is of the format [stance]
    # b_id_to_article = {b_id} -> ['list', 'rep', 'of', 'article', 'tokens']
    # h_id_to_headline = {h_id} -> ['list', 'rep', 'of', 'headline', 'tokens']
    # h_id_b_id_to_stance = {(h_id, b_id)} -> stance
    # raw_article_id_to_b_id = {raw_article_id} -> b_id
    # headline_to_h_id = = {headline_str_rep} -> h_id
    X, y, b_id_to_article, h_id_to_headline, h_id_b_id_to_stance, raw_article_id_to_b_id, headline_to_h_id = compute_splits(training_size, random_split, debug)

    # Get Glove Embeddings
    if glove_set is None:
        word_to_glove_index, glove_matrix = create_glove_dict()
    else:
        word_to_glove_index = glove_set[0]
        glove_matrix = glove_set[1]

    if classification_problem == 1:
        # "agree", "disagree", "discuss", "unrelated" classification problem
        X_train_split = X[0]
        X_dev_split = X[1]
        X_test_split = X[2]
        y_train_split = y[0]
        y_dev_split = y[1]
        y_test_split = y[2]

    elif classification_problem == 2:
        # "related", "unrelated" classifcation problem
        X_train_split, y_train_split = split_by_unrelated_versus_related(X[0], y[0])
        X_dev_split, y_dev_split = split_by_unrelated_versus_related(X[1], y[1])
        X_test_split, y_test_split = split_by_unrelated_versus_related(X[2], y[2])

    elif classification_problem == 3:
        # "agree", "disagree", "discuss" classification problem
        X_train_split, y_train_split = split_by_related_class(X[0], y[0])
        X_dev_split, y_dev_split = split_by_related_class(X[1], y[1])
        X_test_split, y_test_split = split_by_related_class(X[2], y[2])

    else:
        raise Exception('Invalid classifcation problem')

    # Compute Glove Index Vector for Each Headline
    # sample id -> computed glove indices
    h_id_to_glove_index_vector = compute_glove_index_vector(h_id_to_headline, word_to_glove_index, truncate_headlines, max_headline_length)
    b_id_to_glove_index_vector = compute_glove_index_vector(b_id_to_article, word_to_glove_index, truncate_articles, max_article_length)

    # Deterimine max lengths amongst headlines and bodies
    # This information is leveraged by RNNs.
    # TODO: Fix potential bug here -- should we using body length and headline length of train not all data
    # TODO: Discuss w/ John + Saachi
    max_body_length = max([len(index_vec) for (b_id, index_vec) in b_id_to_glove_index_vector.items()])
    max_headline_length = max([len(index_vec) for (h_id, index_vec) in h_id_to_glove_index_vector.items()])
    max_input_lengths = (max_headline_length, max_body_length)

    # Create tuples for train, dev, and test
    # These tuples are (headline_list, article_list, h_seq_lengths, article_seq_lengths)
    X_train_input = compute_glove_id_embeddings(
        X_train_split,
        h_id_to_glove_index_vector,
        b_id_to_glove_index_vector,
        max_input_lengths,
        concatenate=False
    )

    X_dev_input = compute_glove_id_embeddings(
        X_dev_split, 
        h_id_to_glove_index_vector,
        b_id_to_glove_index_vector,
        max_input_lengths,
        concatenate=False
    )

    X_test_input = compute_glove_id_embeddings(
        X_test_split,
        h_id_to_glove_index_vector,
        b_id_to_glove_index_vector,
        max_input_lengths,
        concatenate=False
    )

    if classification_problem == 1:
        # "agree", "disagree", "discuss", "unrelated" classification problem
        y_train_input = compute_stance_embeddings(y_train_split)
        y_dev_input = compute_stance_embeddings(y_dev_split)
        y_test_input = compute_stance_embeddings(y_test_split)

    elif classification_problem == 2:
        # "related", "unrelated" classifcation problem
        y_train_input = compute_stance_embeddings(y_train_split, RELATED_UNRELATED_MAPPING)
        y_dev_input = compute_stance_embeddings(y_dev_split, RELATED_UNRELATED_MAPPING)
        y_test_input = compute_stance_embeddings(y_test_split, RELATED_UNRELATED_MAPPING)

    elif classification_problem == 3:
        # "agree", "disagree", "discuss" classification problem
        y_train_input = compute_stance_embeddings(y_train_split, RELATED_CLASS_MAPPING)
        y_dev_input = compute_stance_embeddings(y_dev_split, RELATED_CLASS_MAPPING)
        y_test_input = compute_stance_embeddings(y_test_split, RELATED_CLASS_MAPPING)

    else:
        raise Exception('Invalid classifcation problem')        

    # Print out meta - info
    if debug_printing:
        with open('./debug/b_id_to_article.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Body ID', "Article"])
            for key, value in b_id_to_article.items():
                writer.writerow([key, ' '.join(value)])

        with open('./debug/h_id_to_headline.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Body ID', "Article"])
            for key, value in h_id_to_headline.items():
                writer.writerow([key, ' '.join(value)])

        with open('./debug/h_id_b_id_to_stance.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Headline ID', "Body ID", "Stance"])
            for key, value in h_id_b_id_to_stance.items():
                writer.writerow([key[0], key[1], value])

        with open('./debug/raw_article_id_to_b_id.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Raw Article ID', "Body ID"])
            for key, value in raw_article_id_to_b_id.items():
                writer.writerow([key, value])

        with open('./debug/X_train_split.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Headline ID', "Body ID", "Stance"])
            for i, t in enumerate(X_train_split):
                writer.writerow([t[0], t[1], y_train_split[i]])

        with open('./debug/X_dev_split.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Headline ID', "Body ID", "Stance"])
            for i, t in enumerate(X_dev_split):
                writer.writerow([t[0], t[1], y_dev_split[i]])

        with open('./debug/X_test_split.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Headline ID', "Body ID", "Stance"])
            for i, t in enumerate(X_test_split):
                writer.writerow([t[0], t[1], y_test_split[i]])   

        with open('./debug/h_id_to_glove_index_vector.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Headline ID', "Glove Index Vector"])
            for key, value in h_id_to_glove_index_vector.items():
                writer.writerow([key] + value.tolist()) 

        with open('./debug/b_id_to_glove_index_vector.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Body ID', "Glove Index Vector"])
            for key, value in b_id_to_glove_index_vector.items():
                writer.writerow([key] + value.tolist()) 

        with open('./debug/word_to_glove_index.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Word', "Glove Index "])
            for key, value in word_to_glove_index.items():
                writer.writerow([key, value])    

    # Consolidate X and y into tuples for easy access
    X = (X_train_input, X_dev_input, X_test_input)
    y = (y_train_input, y_dev_input, y_test_input)
    return X, y, glove_matrix, max_input_lengths, word_to_glove_index

def create_glove_dict():
    """ Read glove data and then create embedding matrix as well as 
        a mapping from word to index in the embedding matrix.
    """
    glove_vectors = read_glove_set() # word to numpy array
    glove_vectors[UNK_TOKEN] = np.zeros((1, GLOVE_SIZE))
    glove_words = glove_vectors.keys()
    word_to_glove_index = {} # mapping into the glove index for embedding
    glove_matrix = np.zeros((len(glove_vectors), GLOVE_SIZE))
    for i, word in enumerate(glove_words):
        word_to_glove_index[word] = i
        glove_matrix[i] = glove_vectors[word]
    return word_to_glove_index, glove_matrix

def read_glove_set():
    """ Read the glove data set and return as a dict from word to numpy array
    """
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

def compute_glove_id_embeddings (id_list, h_id_to_glove_index_vector, b_id_to_glove_index_vector, max_input_lengths, concatenate=False):
    """ Return a matrix/es where each row is an element
        max_input_lengths = (headline length, body length)
        if concanatenate is true,  for each sample concatenate the index vectors of the headline and the body (headline + body)
        otherwise return a tuple (headline_input_matrix, body_input_matrix)
    """
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
        headline_seq_lengths = []
        body_seq_lengths = []

        for index, (h_id, b_id) in enumerate(id_list):
            h_index_vector = h_id_to_glove_index_vector[h_id]
            b_index_vector = b_id_to_glove_index_vector[b_id]
            headline_length = len(h_index_vector)
            body_length = len(b_index_vector)

            headline_list[index][:headline_length] = h_index_vector
            body_list[index][:body_length] = b_index_vector
            headline_seq_lengths.append(headline_length)
            body_seq_lengths.append(body_length)

        return (headline_list, body_list, headline_seq_lengths, body_seq_lengths)

def compute_glove_index_vector (id_to_text, word_to_glove_index, truncate=True, truncate_size=500):
    """ id_to_text should be {id -> text} for either headline or body
        word_to_glove_index should be {word -> word's index in glove}

        Returns sample_id_to_glove_index_vector = {id} -> np.
    """
    sample_id_to_glove_index_vector = {}
    for (sample_id, text) in id_to_text.items():
        if truncate:
            trunc_text = text[:truncate_size]
        else:
            trunc_text = text
        index_vector = np.zeros(len(trunc_text))
        for i, word in enumerate(trunc_text):
            if word not in word_to_glove_index:
                word = UNK_TOKEN
            index_vector[i] = word_to_glove_index[word]
        sample_id_to_glove_index_vector[sample_id] = index_vector
    return sample_id_to_glove_index_vector

def compute_stance_embeddings(stance_list, mapping=LABEL_MAPPING):
    """ Create matix of 1-hot representations of tance labels.
    """
    labels_matrix = np.zeros((len(stance_list), len(mapping)))
    for i in range(0,len(stance_list)):
        labels_matrix[i][stance_list[i]] = 1
    return labels_matrix

if __name__ == '__main__':
    X, y, glove_matrix, max_input_lengths, word_to_glove_index = create_embeddings(
        training_size=1.0,
        random_split=True,
        truncate_headlines=False,
        truncate_articles=True,
        classification_problem=3,
        max_headline_length=500,
        max_article_length=500,
        glove_set=None,
        debug_printing=True,
        # debug=False,
        # debug=True,
    )

# -----------------------------------------------
# LEGACY CODE -- THIS IS MANTAINED FOR BASIC LSTM
# -----------------------------------------------

def create_inputs_by_glove(concatenate=True, truncate=True):
    """ if concatenate is true then X's are one input matrix which have article and headline concatenated
        otherwise return tuple of input matrices for each input x
        Note: num_sentences_to_keep only utilized if truncate = False
    """
    # X is [(headline id, body id)]
    X, y, b_id_to_article, h_id_to_headline, h_id_b_id_to_stance, raw_article_id_to_b_id, headline_to_h_id = compute_splits(TRAINING_SIZE, USE_RANDOM_FNC)

    if (UNDER_REPRESENT):
        X_train, y_train = underRepresent(X_train, y_train, PERC_UNRELATED)

    # read glove
    word_to_glove_index, glove_matrix = create_glove_dict()

    # compute glove index vector for every headline
    # sample id -> computed glove indices
    h_id_to_glove_index_vector = compute_glove_index_vector(h_id_to_headline, word_to_glove_index, truncate)
    b_id_to_glove_index_vector = compute_glove_index_vector(b_id_to_article, word_to_glove_index, truncate, MAX_BODY_LENGTH)
    max_body_length = max([len(index_vec) for (b_id, index_vec) in b_id_to_glove_index_vector.items()])
    max_headline_length = max([len(index_vec) for (h_id, index_vec) in h_id_to_glove_index_vector.items()])
    max_input_lengths = (max_headline_length, max_body_length)

    # Compute embeddings for headlines and articles
    X_train_input = compute_glove_id_embeddings(X[0], h_id_to_glove_index_vector, b_id_to_glove_index_vector, max_input_lengths, concatenate)
    X_dev_input = compute_glove_id_embeddings(X[1], h_id_to_glove_index_vector, b_id_to_glove_index_vector, max_input_lengths, concatenate)
    X_test_input = compute_glove_id_embeddings(X[2], h_id_to_glove_index_vector, b_id_to_glove_index_vector, max_input_lengths, concatenate)

    y_train_input = compute_stance_embeddings(y[0])
    y_dev_input = compute_stance_embeddings(y[1])
    y_test_input = compute_stance_embeddings(y[2])

    return X_train_input, X_dev_input, X_test_input, y_train_input, y_dev_input, y_test_input, glove_matrix, max_input_lengths

# --------------------------------
# Code for Baseline Neural Network
# --------------------------------

def save_glove_sums_matrix(id_to_body, id_to_headline, glove_vectors):
    """ Read the dicts id to body and id to headline.
        For each example, compute the sum of the glove vectors
        for each word. Create two numpy matrices to store
        the sums per example and save as binaries
    """
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

def construct_glove_sum_binaries():
    """ Construct binaries where each text is represented as the sum of the glove 
        vectors for the words.
    """
    b_id_to_body, h_id_to_headline, h_id_b_id_to_stance = construct_data_set()
    glove_vectors = read_glove_set()
    save_glove_sums_matrix(b_id_to_body, h_id_to_headline, glove_vectors)
    write_id_id_stance(h_id_b_id_to_stance)

def read_glove_sum_binaries():
    """ Read the binaries where each text is represented as the sum of the glove vectors for the words
    """
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
