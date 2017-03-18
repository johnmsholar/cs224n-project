import pickle
import json
import os
from ast import literal_eval
import argparse

filenames = [
    'bleu.pkl',
    'overlap_clean.pkl',
    'tfidf.pkl',
    'overlap.pkl',
    'tfidf_clean.pkl'
]

def prepare_json_format(d):
    return dict((str(key), value) for key, value in d.items())

def retrieve_json_format(d):
    return dict((literal_eval(key), value) for key, value in d.items())

parser = argparse.ArgumentParser()
parser.add_argument('--file')
args = parser.parse_args()
filename = args.file

raw_filename = os.path.splitext(filename)[0]
output_filename = raw_filename + '.json'
with open(filename, 'r') as infile:
    data = pickle.load(infile)
data = prepare_json_format(data)
with open(output_filename, 'w+') as outfile:
    json.dump(data, outfile)

