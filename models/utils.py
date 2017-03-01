from sklearn.model_selection import train_test_split
import numpy as np

from enum import Enum
import csv

GLOVE_SIZE = 300
# Enums
class Labels(Enum):
    UNRELATED = 0
    DISCUSS = 1
    AGREE = 2
    DISAGREE = 3

# Constants
LABEL_MAPPING = {
    'unrelated': Labels.UNRELATED,
    'discuss': Labels.DISCUSS,
    'agree': Labels.AGREE,
    'disagree': Labels.DISAGREE,
}

RANDOM_STATE = 42
TEST_SIZE = .20
>>>>>>> 388852f0326e713731e8cd2099833efb40808387

def construct_data_set():
    # File Headers
    body_id_header = 'Body ID'
    article_body_header = 'articleBody'
    headline_header = 'Headline'
    stance_header = 'Stance'

    # Mappings
    b_id_to_body = {}
    h_id_to_headline = {}
    h_id_b_id_to_stance = {}

    # Read Article Bodies
    with open('../fnc_1/train_bodies.csv') as bodies_file:
        bodies_reader = csv.DictReader(bodies_file, delimiter = ',')
        for row in bodies_reader:
            b_id = int(row[body_id_header])
            article_body = row[article_body_header]
            b_id_to_body[b_id] = article_body

    # Read Headline, ID -> Stance Mappings
    with open('../fnc_1/train_stances.csv') as stances_file:
        stances_reader = csv.DictReader(stances_file, delimiter = ',')
        for h_id, row in enumerate(stances_reader):
            headline = row[headline_header]
            b_id = int(row[body_id_header])
            h_id_to_headline[h_id] = headline
            h_id_b_id_to_stance[(h_id, b_id)] = LABEL_MAPPING[row[stance_header]]

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

def test_train_split(data, test_size):
    # Data is a dict of (headline_id, body_id) -> stance 
    X  = data.keys()
    y = [data[x] for x in X]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    b_id_to_body, h_id_to_headline, h_id_b_id_to_stance, X_train, X_test, y_train, y_test = construct_data_set()

