import random
import os
from collections import defaultdict
import filenames
import csv
from __init__ import LABEL_MAPPING
# Train/Test Split as defined by FNC-1 Baseline

rgen = random.Random()
rgen.seed(1489215)

# id_id_stance should be (h_id b_id)-> stance
def compute_splits(id_id_stance, training=0.8, random=True):
    num_articles = len(set([ids[1] for (ids, stance) in id_id_stance.items()]))
    if (random):
        training_ids, hold_out_ids = generate_random_hold_out_split(num_articles, training)
    else:
        training_ids, hold_out_ids = generate_original_holdouts()

    rgen.shuffle(training_ids)
    train_ids = training_ids[:int(training * len(training_ids))]
    dev_ids = training_ids[int(training * len(training_ids)):]

    train_ids = set(train_ids)
    dev_ids = set(dev_ids)
    # train pairs are (headline, body)
    x_train = []
    y_train = []
    x_dev = []
    y_dev = []
    x_test = []
    y_test = []
    for (id_pair, stance) in id_id_stance.items():
        if id_pair[1] in train_ids:
            x_train.append(id_pair)
            y_train.append(stance)
        elif id_pair[1] in dev_ids:
            x_dev.append(id_pair)
            y_dev.append(stance)
        else:
            x_test.append(id_pair)
            y_test.append(stance)
    print "Train: {}, Dev: {}, Test: {}".format(len(x_train), len(x_dev), len(x_test))
    return x_train, x_dev, x_test, y_train, y_dev, y_test

# id_pairs are h_id, b_id
# cut the unrelated samples until there are < perc_unrelated samples in the whole set
def underRepresent(id_pairs, stances, perc_unrelated):
    by_stance = distribute_by_stance(id_pairs, stances)
    rgen.shuffle(by_stance[0])
    curr_num_unrelated = len(by_stance[LABEL_MAPPING["unrelated"]])
    num_total = len(id_pairs)
    needed_num_unrelated = int(perc_unrelated * num_total)
    if (needed_num_unrelated < curr_num_unrelated):
        by_stance[0] = by_stance[0][:needed_num_unrelated]
    # (id_pair, stance)
    new_num_unrelated = len(by_stance[0])
    new_id_pairs_stance = []
    for (stance, stance_id_pair_list) in by_stance.items():
        new_id_pairs_stance += [(id_pair, stance) for id_pair in stance_id_pair_list]
    rgen.shuffle(new_id_pairs_stance)
    id_pairs = [sample[0] for sample in new_id_pairs_stance]
    stances = [sample[1] for sample in new_id_pairs_stance]
    print "Under-representing, so now {} samples from {}".format(new_num_unrelated, curr_num_unrelated)
    return id_pairs, stances

# return a dict with {stance -> (h_id, b_id)}
def distribute_by_stance(id_pairs, stances):
    by_stance = {i: [] for i in range(0, 4)}
    assert len(id_pairs) == len(stances)
    for (i, id_pair) in enumerate(id_pairs):
        stance = stances[i]
        by_stance[stance].append(id_pair)
    return by_stance


# returns a list of article ids for training and for hold out from original 
def generate_original_holdouts():
    training_fnc_ids = read_list_of_ids(filenames.ORIG_TRAIN_BODY_ID_FNAME)
    hold_out_fnc_ids = read_list_of_ids(filenames.ORIG_HOLDOUT_BODY_ID_FNAME)
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
    article_ids = [i for i in range(0, num_articles)] # article ids are consecutive
    rgen.shuffle(article_ids)  # and shuffle that list
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
