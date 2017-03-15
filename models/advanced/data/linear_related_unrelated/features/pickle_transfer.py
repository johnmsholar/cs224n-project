import pickle
import json
import os

filenames = [
    'bleu.pkl',
    'headline_gram.pkl',
    'overlap_clean.pkl',
    'tfidf.pkl',
    'cross_gram_count.pkl'
    'overlap.pkl',
    'tfidf_clean.pkl'
]

for index, filename in enumerate(filenames):
    print 'Evaluating file {0} ({1} of {2})'.format(
        filename, index + 1, len(filenames))
    raw_filename = os.path.splittext(filename)[0]
    output_filename = os.path.join(raw_filename, '.json')
    with open(filename, 'r') as infile:
        data = pickle.load(file)
    with open(output_filename, 'w+') as outfile:
        json.dump(data, outfile)
