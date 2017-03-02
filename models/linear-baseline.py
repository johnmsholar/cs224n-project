import sklearn
import nltk
import itertools
from utils import construct_data_set

# Generate modified BLEU scores for each (healdine, article) pair, in which BLEU
# score is evaluated for a series of overlapping segments of the article.
# See description below for more information.
def generate_bleu_score_features(b_id_to_body, h_id_to_headline, h_id_b_id_to_stance):

    # Slice the body text into overlapping segments of length BLEU_SEGMENT_LENGTH,
    #   such that each segment overlaps with half of the next segment (to ensure that
    #   no phrase shorter than BLEU_SEGMENT_LENGTH is scliced in two
    # For every segment, compute the BLEU score of the headline with respect to the segment,
    #   and return the maximum such score
    # Ideally, this allows us to examine if at any point, the short headline matches a
    #   segment of the much longer reference text, while avoiding the brevity penalty
    #   associated with comparing a short hypothesis to a long reference text.
    def max_segment_bleu_score(head, body):
        BLEU_SEGMENT_LENGTH = 40
        SEGMENT_INCREMENT = BLEU_SEGMENT_LENGTH / 2
        max_bleu_score = 0.0
        for start_index in range(0, len(body), SEGMENT_INCREMENT):
            body_segment = body[start_index:start_index + BLEU_SEGMENT_LENGTH]
            score = nltk.translate.bleu_score.sentence_bleu(body_segment, head)
            max_bleu_score = max(score, max_bleu_score)
        return max_bleu_score

    num_pairs = len(h_id_b_id_to_stance.keys())
    print('Processing BLEU Score Feature for {0} pairs'.format(num_pairs))
    bleu_score_feature_vectors = {}
    BLEU_FEATURE_NAME = 'max_segment_bleu_score'
    for index, (h_id, b_id) in enumerate(h_id_b_id_to_stance):
        if index % 100 == 0:
            print(float(index) / num_pairs)
        headline = h_id_to_headline[h_id]
        body = b_id_to_body[b_id]
        score = max_segment_bleu_score(headline, body)
        bleu_score_feature_vectors[(h_id, b_id)] = {BLEU_FEATURE_NAME : score}
    return bleu_score_feature_vectors

# Not yet implemented
def generate_overlap_features(b_id_to_body, h_id_to_headline, h_id_b_id_to_stance):
    pass

# Generates indicator features for all unigrams and bigrams in the headline
def generate_headline_gram_features(b_id_to_body, h_id_to_headline, h_id_b_id_to_stance):
    HEADLINE_GRAM_FEATURE_NAME = 'headline_gram'
    headline_gram_features = {}
    for (h_id, b_id) in h_id_b_id_to_stance:
        headline = h_id_to_headline[h_id]
        headline_unigrams = nltk.ngrams(headline, 1)
        headline_bigrams = nltk.ngrams(headline, 2)
        pair_headline_gram_features = {}
        for gram in headline_unigrams + headline_bigrams:
            pair_headline_gram_features[(HEADLINE_GRAM_FEATURE_NAME, gram)] = 1.0
        headline_gram_features[(h_id, b_id)] = pair_headline_gram_features
    return headline_gram_features

# For every (headline, article) pair, Generate a feature vector containing indicator features for:
    #   1. Cross-Unigrams: every pair of words across the headline and article which share a POS tag
    #   2. Cross-Bigrams: every pair of bigrams across the healdine and article which share a POS tag on the second word
def generate_cross_gram_features(b_id_to_body, h_id_to_headline, h_id_b_id_to_stance):

    # For a single (headline, article) pair, generate a single feature vector composed of all cross-ngrams
    # matching the conditions described above
    def single_pair_cross_ngram_features(headline, article, n):
        CROSS_NGRAM_FEATURE_NAME = 'cross_ngram'
        headline_unigrams = nltk.ngrams(headline, n)
        article_unigrams = nltk.ngrams(article, n)
        headline_pos = nltk.pos_tag(headline)
        article_pos = nltk.pos_tag(article)
        all_pos = headline_pos + article_pos
        unique_pos_classes = set([token_pos[1] for token_pos in all_pos])
        single_pair_cross_ngram_features = {}
        for pos_class in unique_pos_classes:
            headline_matches = [g for i, g in enumerate(headline_unigrams) if headline_pos[i + n - 1] == pos_class]
            article_matches = [g for i, g in enumerate(article_unigrams) if article_pos[i + n - 1] == pos_class]
            for cross_gram in itertools.product([headline_matches, article_matches]):
                single_pair_cross_ngram_features[(CROSS_NGRAM_FEATURE_NAME, cross_gram)] = 1.0
        return single_pair_cross_ngram_features

    all_cross_gram_features = {}
    for (h_id, b_id) in h_id_b_id_to_stance:
        headline = h_id_to_headline[h_id]
        body = b_id_to_body[b_id]
        unigram_features = single_pair_cross_ngram_features(headline, body, 1)
        bigram_features = single_pair_cross_ngram_features(headline, body, 2)
        gram_features = dict(unigram_features.items() + bigram_features.items())
        all_cross_gram_features[(h_id, b_id)] = gram_features
    return all_cross_gram_features

def join_features_on_key(feature_maps, h_id_b_id_to_stance):
    all_keys_aggregated_features_dict = {}
    for (h_id, b_id) in h_id_b_id_to_stance:
        key = (h_id, b_id)
        key_feature_vectors = [mapping[key] for mapping in feature_maps]
        aggregated_feature_vector = reduce(lambda x, y: x + y, key_feature_vectors)
        aggregated_features_dict = dict(aggregated_feature_vector)
        all_keys_aggregated_features_dict[key] = aggregated_features_dict
    return all_keys_aggregated_features_dict

def main():
    b_id_to_body, h_id_to_headline, h_id_b_id_to_stance = construct_data_set()
    print('DATASET CONSTRUCTED')

    bleu_score_features = generate_bleu_score_features(b_id_to_body, h_id_to_headline, h_id_b_id_to_stance)
    headline_gram_features = generate_headline_gram_features(b_id_to_body, h_id_to_headline, h_id_b_id_to_stance)
    cross_gram_features = generate_cross_gram_features(b_id_to_body, h_id_to_headline, h_id_b_id_to_stance)
    print('INDIVIDUAL FEATURE VECTORS GENERATED')

    feature_maps = [bleu_score_features, headline_gram_features, cross_gram_features]
    all_keys_aggregated_features_dict = join_features_on_key(feature_maps, h_id_b_id_to_stance)
    print('GLOBAL FEATURE VECTORS GENERATED')

if __name__ == '__main__':
    main()
