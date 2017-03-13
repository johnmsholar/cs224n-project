#Adapted from https://github.com/FakeNewsChallenge/fnc-1/blob/master/scorer.py
#Original credit - @bgalbraith

import sklearn
import sys
sys.path.insert(0, '../')
from util import plot_confusion_matrix, save_confusion_matrix
LABELS = [0,1,2,3]
LABELS_HEADER = ['agree', 'disagree', 'discuss', 'unrelated']
LABELS_RELATED = [0,1]
LABELS_RELATED_HEADER = ['unrelated','related']
RELATED = LABELS[0:3]

def score_submission(gold_labels, test_labels):
    score = 0.0
    cm = [[0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0]]

    for i, (g, t) in enumerate(zip(gold_labels, test_labels)):
        g_stance, t_stance = g, t
        if g_stance == t_stance:
            score += 0.25
            if g_stance != 'unrelated':
                score += 0.50
        if g_stance in RELATED and t_stance in RELATED:
            score += 0.25

        cm[LABELS.index(g_stance)][LABELS.index(t_stance)] += 1

    return score, cm


def print_confusion_matrix(cm):
    lines = []
    header = "\n|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format('', *LABELS_HEADER)
    line_len = len(header)
    lines.append("-"*line_len)
    lines.append(header)
    lines.append("-"*line_len)

    hit = 0
    total = 0
    for i, row in enumerate(cm):
        hit += row[i]
        total += sum(row)
        lines.append("|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format(LABELS_HEADER[i],
                                                                   *row))
        lines.append("-"*line_len)
    print('\n'.join(lines))
    return '\n'.join(lines)

def pretty_report_score(actual, preds):
    cm = sklearn.metrics.confusion_matrix(actual, preds)
    classes = ['AGREE', 'DISAGREE', 'DISCUSS', 'UNRELATED']
    save_confusion_matrix(cm, classes, filename, normalize=True)
    test_score = report_score(actual, preds)
    return test_score


def report_score(actual,predicted):
    score,cm = score_submission(actual,predicted)
    best_score, _ = score_submission(actual,actual)

    lines = print_confusion_matrix(cm)
    print("Score: " +str(score) + " out of " + str(best_score) + "\t("+str(score*100/best_score) + "%)")
    lines += '\n' + ("Score: " +str(score) + " out of " + str(best_score) + "\t("+str(score*100/best_score) + "%)")
    return score*100/best_score, lines


if __name__ == "__main__":
    actual = [0,0,0,0,1,1,0,3,3]
    predicted = [0,0,0,0,1,1,2,3,3]

    report_score([LABELS[e] for e in actual],[LABELS[e] for e in predicted])