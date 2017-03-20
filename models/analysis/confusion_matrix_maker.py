import numpy as np
import sys

sys.path.insert(0, '../../')
from models.util import save_confusion_matrix

if __name__ == '__main__':
    data = np.array(
        [[444, 30, 181],
        [53, 43, 65],
        [214, 33, 1726]]
        )
    classes = ['Agree', 'Disagree', 'Discuss']
    title = 'Bidirec. C.E. LSTMs w. Bidrec. Global Attention (Suboptimal F1 Score)'
    filename = '../advanced/data/bidirectional_attention_bidirectional_conditional_lstm/plots/suboptimal_cm.png'

    save_confusion_matrix(data, classes, filename, normalize=True, title='')

    # data = np.array(
    #     [[0.9427, .0573],
    #     [.0066, .9934]]
    #     )
    # classes = ['Related', 'Unrelated']
    # title = 'Linear SVM Classifer wiht TF-IDF Cosine Similarity Feat'
    # filename = '../advanced/data/linear_related_unrelated/plots/cm.png'
    # save_confusion_matrix(data, classes, filename, normalize=False, title=title)

    data = np.array(
        [[475, 0, 180],
        [86, 0, 75],
        [214, 0, 1759]]
        )
    classes = ['Agree', 'Disagree', 'Discuss']
    title = 'Bidirec. C.E. LSTMs w. Bidrec. Global Attention (Optimal F1 Score)'
    filename = '../advanced/data/bidirectional_attention_bidirectional_conditional_lstm/plots/optimal_cm.png'

    save_confusion_matrix(data, classes, filename, normalize=True, title='')
