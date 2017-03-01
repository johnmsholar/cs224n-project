from sklearn.model_selection import train_test_split
import numpy as np
import csv

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



def test_train_split():


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

