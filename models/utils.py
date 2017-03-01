from sklearn.model_selection import train_test_split
import numpy as np

from enum import Enum
import csv
import re
import string

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
SPACE_CHAR = ' '
NEWLINE_CHAR = '\n'
DASH_CHAR = '-'

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
            article = clean(article_body)
            b_id_to_body[b_id] = article

    # Read Headline, ID -> Stance Mappings
    with open('../fnc_1/train_stances.csv') as stances_file:
        stances_reader = csv.DictReader(stances_file, delimiter = ',')
        for h_id, row in enumerate(stances_reader):
            headline = row[headline_header]
            b_id = int(row[body_id_header])
            h_id_to_headline[h_id] = headline
            h_id_b_id_to_stance[(h_id, b_id)] = LABEL_MAPPING[row[stance_header]]

    # Split train/test data
    X_train, X_test, y_train, y_test = test_train_split(h_id_b_id_to_stance, TEST_SIZE)

    return b_id_to_body, h_id_to_headline, h_id_b_id_to_stance, X_train, X_test, y_train, y_test

def clean(article_body):
    article_body = article_body.replace(NEWLINE_CHAR, SPACE_CHAR)
    article_body = article_body.replace(DASH_CHAR, SPACE_CHAR)    

    def clean_word(word):
        w = word.lower()
        tokens = re.findall(r"[\w']+|[.,!?;]", w)
        # w = w.translate(None, string.punctuation)
        return [t.strip() for t in tokens if (t.isalnum() or t in string.punctuation) and t.strip() != '']

    cleaned_article = []
    for w in str.split(article_body, SPACE_CHAR):
        c_word = clean_word(w)
        if c_word is not SPACE_CHAR:
            cleaned_article.extend(c_word)

    return cleaned_article

def test_train_split(data, test_size):
    # Data is a dict of (headline_id, body_id) -> stance 
    X  = data.keys()
    y = [data[x] for x in X]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    b_id_to_body, h_id_to_headline, h_id_b_id_to_stance, X_train, X_test, y_train, y_test = construct_data_set()
    print b_id_to_body
