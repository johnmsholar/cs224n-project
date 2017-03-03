import random
import os
from collections import defaultdict
import filenames
import csv
# Train/Test Split as defined by FNC-1 Baseline



def compute_splits(id_id_stance, training=0.8, random=True):
    num_articles = len(set([ids[0] for (ids, stance) in id_id_stance.items()]))
    if (random):
        training_ids, hold_out_ids = generate_random_hold_out_split(num_articles)
    else:
        training_ids, hold_out_ids = generate_original_holdouts()
    training_ids = set(training_ids)
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for (id_pair, stance) in id_id_stance.items():
        if id_pair[1] in training_ids:
            x_train.append(id_pair)
            y_train.append(stance)
        else:
            x_test.append(id_pair)
            y_test.append(stance)
    return x_train, x_test, y_train, y_test

# returns a list of article ids for training and for hold out from original 
def generate_original_holdouts():
    training_fnc_ids = read_list_of_ids(filenames.ORIG_TRAIN_A_ID_FNAME)
    hold_out_fnc_ids = read_list_of_ids(filenames.ORIG_TRAIN_H_ID_FNAME)
    b_id_to_index = {}
    # Read Article Bodies and get a map of id to index
    with open(filenames.TRAIN_BODIES_FNAME) as bodies_file:
        bodies_reader = csv.DictReader(bodies_file, delimiter = ',')
        index = 0
        for row in bodies_reader:
            b_id = int(row['Body ID'])
            b_id_to_index[b_id] = index
            index+=1
    # convert from FNC IDs to index IDs
    training_fnc_ids = [b_id_to_index[a_id] for a_id in training_fnc_ids]
    hold_out_fnc_ids = [b_id_to_index[a_id] for a_id in hold_out_fnc_ids]
    return training_fnc_ids, hold_out_fnc_ids

# given a file with an ID on every line, return the list of IDs
def read_list_of_ids(filename):
    id_list = []
    with open(filename, 'rb') as csvfile:
        id_reader = csv.reader(csvfile, delimiter = ' ', quotechar='|')
        for l in id_reader:
            id_list.append(int(l[0]))
    return id_list

# generate random article split
# pass in the number of articles
def generate_random_hold_out_split (num_articles, training = 0.8):
    r = random.Random()
    r.seed(1489215)
    article_ids = [i for i in range(0, num_articles)] # article ids are consecutive
    r.shuffle(article_ids)  # and shuffle that list
    training_ids = article_ids[:int(training * len(article_ids))]
    hold_out_ids = article_ids[int(training * len(article_ids)):]
    return training_ids, hold_out_ids

def read_ids(file,base):
    ids = []
    with open(base+"/"+file,"r") as f:
        for line in f:
           ids.append(int(line))
        return ids


def kfold_split(dataset, training = 0.8, n_folds = 10, base_dir="splits"):
    if not (os.path.exists(base_dir+ "/"+ "training_ids.txt")
            and os.path.exists(base_dir+ "/"+ "hold_out_ids.txt")):
        generate_hold_out_split(dataset,training,base_dir)

    training_ids = read_ids("training_ids.txt", base_dir)
    hold_out_ids = read_ids("hold_out_ids.txt", base_dir)

    folds = []
    for k in range(n_folds):
        folds.append(training_ids[int(k*len(training_ids)/n_folds):int((k+1)*len(training_ids)/n_folds)])

    return folds,hold_out_ids


def get_stances_for_folds(dataset,folds,hold_out):
    stances_folds = defaultdict(list)
    stances_hold_out = []
    for stance in dataset.stances:
        if stance['Body ID'] in hold_out:
            stances_hold_out.append(stance)
        else:
            fold_id = 0
            for fold in folds:
                if stance['Body ID'] in fold:
                    stances_folds[fold_id].append(stance)
                fold_id += 1

    return stances_folds,stances_hold_out
