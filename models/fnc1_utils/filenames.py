# Raw training data from FNC-1 Challenge
TRAIN_BODIES_FNAME = "../../fnc_1/train_bodies.csv"
TRAIN_STANCES_FNAME = "../../fnc_1/train_stances.csv"


# Word Vector Representations
GLOVE_FILENAME = "../../../glove.6B/glove.6B.300d.txt"

# Output embeddings and embedding references
BODY_EMBEDDING_FNAME = "../../fnc_1/glove_body_matrix"
HEADLINE_EMBEDDING_FNAME = "../../fnc_1/glove_headline_matrix"
ID_ID_STANCES_FNAME = "../../fnc_1/id_id_stance.csv"

# Train/Test Splits as defined by FNC-1 Baseline
# utilizing our indexing scheme.
TRAINING_SPLIT_FNAME = '../../splits/custom_training_ids.txt'
TEST_SPLIT_FNAME = '../../splits/custom_hold_out_ids.txt'

# location of original baseline splits
ORIG_TRAIN_BODY_ID_FNAME = '../../splits/training_ids.txt'
ORIG_HOLDOUT_BODY_ID_FNAME = '../../splits/hold_out_ids.txt'
