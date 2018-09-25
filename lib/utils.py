import numpy as np
from sklearn.metrics import average_precision_score


def load_data(data_path):
    """load array data from data_path"""
    data = np.load(data_path)
    return data['X_train'], data['y_train'], data['X_test'], data['y_test']


def calculate_average_precision(label, index, similarity, num_search_sample):
    """calculate average precision of similar search result.
    The average precison is calculated over num_search_sample
    """
    label_idx = np.array([label[idx] for idx in index])
    label_idx_true = np.array([np.where(row == row[0], 1, 0) for row in label_idx])

    label_idx_true = label_idx_true[:, 1:]

    ap = []
    for i in range(num_search_sample):
        ap.append(average_precision_score(label_idx_true[i], similarity[i]))

    return ap
