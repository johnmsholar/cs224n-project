import numpy as np
import sys

sys.path.insert(0, '../../')
from models.util import save_confusion_matrix

if __name__ == '__main__':
    # data = np.array(
    #     [[444, 30, 181],
    #     [53, 43, 65],
    #     [214, 33, 1726]]
    #     )
    # classes = ['Agree', 'Disagree', 'Discuss']
    # title = 'Bidirec. C.E. LSTMs w. Bidrec. Global Attention (Suboptimal F1 Score)'
    # filename = '../advanced/data/bidirectional_attention_bidirectional_conditional_lstm/plots/suboptimal_cm.png'

    # save_confusion_matrix(data, classes, filename, normalize=True, title='')

    # data = np.array(
    #     [[0.9427, .0573],
    #     [.0066, .9934]]
    #     )
    # classes = ['Related', 'Unrelated']
    # title = 'Linear SVM Classifer wiht TF-IDF Cosine Similarity Feat'
    # filename = '../advanced/data/linear_related_unrelated/plots/cm.png'
    # save_confusion_matrix(data, classes, filename, normalize=False, title=title)

    # data = np.array(
    #     [[475, 0, 180],
    #     [86, 0, 75],
    #     [214, 0, 1759]]
    #     )
    # classes = ['Agree', 'Disagree', 'Discuss']
    # title = 'Bidirec. C.E. LSTMs w. Bidrec. Global Attention (Optimal F1 Score)'
    # filename = '../advanced/data/bidirectional_attention_bidirectional_conditional_lstm/plots/optimal_cm.png'

    # save_confusion_matrix(data, classes, filename, normalize=True, title='')

    # data = np.array(
    #     [[431, 13, 211],
    #     [56, 17, 88],
    #     [296, 8, 1669]]
    #     )
    # classes = ['Agree', 'Disagree', 'Discuss']
    # title = 'Bilateral Multi-Perspective Matching Model'
    # filename = '../advanced/data/bimpmp/plots/cm.png'

    # save_confusion_matrix(data, classes, filename, normalize=True, title='')


    # data = np.array(
    #     [[318, 36, 200, 101],
    #     [34, 45, 57, 25],
    #     [208, 18, 1579, 177],
    #     [176, 44, 608, 6625]]
    #     )
    # classes = ['Agree', 'Disagree', 'Discuss', 'Unrelated']
    # title = ''
    # filename = '../baselines/data/plots/snli_cm.png'
    # save_confusion_matrix(data, classes, filename, normalize=True, title='')

    data = np.array(
        [[0, 1, 0, 654],
        [0, 0, 0, 161],
        [0, 0, 0, 1973],
        [0, 1, 0, 7453]]
        )
    classes = ['Agree', 'Disagree', 'Discuss', 'Unrelated']
    title = ''
    filename = '../baselines/data/plots/lstm_concat.png'
    save_confusion_matrix(data, classes, filename, normalize=True, title='')
