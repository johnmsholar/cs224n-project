from models.fnc1_utils.featurizer import construct_data_set
import matplotlib.pyplot as plt
import numpy as np

ARTICLE_HISTOGRAM_FNAME = 'article_histogram.png'
HEADLINE_HISTOGRAM_FNAME = 'headline_histogram.png'

def plot_histogram(filename, data, bins, title, xlabel='Value', ylabel='Count'):
    plt.hist(data, bins=bins, histtype='bar', rwidth=.75)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def within_standard_dev(dataset, standard_deviations):
    dataset = sorted(dataset)
    std = np.std(dataset)
    mean = np.mean(dataset)
    lower_bound = mean - standard_deviations * std
    upper_bound = mean + standard_deviations * std
    return filter(lambda x: x >= lower_bound and x <= upper_bound, dataset)


def main():
    b_id_to_body, h_id_to_headline, _ = construct_data_set()
    article_lengths = [len(value) for key, value in b_id_to_body.items()]
    headline_lengths = [len(value) for key, value in h_id_to_headline.items()]
    ARTICLE_FNAME_ROOT = 'article_histogram_{0}_std.png'
    HEADLINE_FNAME_ROOT = 'headline_histogram_{0}_std.png'
    for i in range(1, 4):
        within_std_article_lengths = within_standard_dev(article_lengths, i)
        within_std_headline_lengths = within_standard_dev(headline_lengths, i)
        article_fname = ARTICLE_FNAME_ROOT.format(i)
        headline_fname = HEADLINE_FNAME_ROOT.format(i)
        plot_histogram(article_fname, within_std_article_lengths, 20, 'Article Lengths')
        plot_histogram(headline_fname, within_std_headline_lengths, 20, 'Headline Lengths')
    plot_histogram(ARTICLE_HISTOGRAM_FNAME, article_lengths, 20, 'Article Lengths')
    plot_histogram(HEADLINE_HISTOGRAM_FNAME, headline_lengths, 20, 'Headline Lengths')


if __name__ == '__main__':
    main()