from sklearn.model_selection import train_test_split
import numpy as np
import csv

GLOVE_SIZE = 300

def construct_data_set():
    # File Headers
    body_id_header = 'Body ID'
    article_body_header = 'articleBody'
    headline_header = 'Headline'

    id_to_body = {}
    id_to_article = {}

    with open('../fnc_1/train_bodies.csv') as bodies_file:
        bodies_reader = csv.DictReader(bodies_file, delimiter = ',')
        for row in bodies_reader:
            id_to_body[row[body_id_header]] = 


    with open('../fnc_1/train_stances.csv') as stances_file:
        stances_reader = csv.reader(stances_file, delimiter = ',')

def read_glove_set():
    default = np.zeros(GLOVE_SIZE)
    glove_vectors = defaultdict(default)
    id_to_glove_body = {}
    with open("../glove.6B/glove.6B.300d.txt") as glove_files:
        glove_reader = csv.DictReader(glove_files, delimiter = ' ')
        for row in glove_reader:
            word = row[0]
            vec = np.array(row[1:])
            glove_vector[word] = vec
    return glove_vectors

def save_glove_sums_matrix(
    glove_vectors, body_fname="glove_body_matrix.txt",
    headline_fname="glove_headline_matrix.txt"):
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
        glove_headline_matrix[body_id] = headline_sum
    # saves as np binaries glove_body | glove_headline
    np.save(body_fname, glove_body_matrix)
    np.save(headline_fname, glove_headline_matrix)

def read_glove_sums(
    body_fname="glove_body_matrix.txt",
    headline_fname="glove_headline_matrix.txt"):

    glove_body_matrix = np.load(body_fname)
    glove_headline_matrix = np.load(headline_fname)
    return glove_body_matrix, glove_headline_matrix

def compute_word_embeddings():
    headline



def test_train_split():


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

