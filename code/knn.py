import time

import numpy as np
import operator

from dtw import dtw
from scipy.spatial.distance import euclidean

from adapdtw import cdtw
from ewkm import partition_abstr


def knn_dist(x_train, y_train, x_test, k, measure, w):
    """
    kNN classifier with DTW
    :param x_train: training set
    :param y_train: training labels
    :param x_test: testing set
    :param k: the number of nearest neighbors
    :param measure: distance measures
    :param w: window_size
    :return: predicted testing labels
    """
    predict_result = []
    train_size = len(x_train)
    test_size = len(x_test)
    dist = np.zeros(train_size)

    for i in range(test_size):
        for j in range(train_size):
            if measure == 'euclidean':
                dist[j] = euclidean(x_test[i], x_train[j])
            elif measure == 'dtw':
                dist[j], cost, acc, path = dtw(x_test[i], x_train[j], euclidean, w=w, s=1)

        sorted_dist = np.argsort(dist)
        class_count = {}

        # count the classes of k nearest neighbors
        for x in range(k):
            sort_label = y_train[sorted_dist[x]]
            class_count[sort_label] = class_count.get(sort_label, 0) + 1

        # get the predicted label from the sorted class number
        sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
        predict_result.append(sorted_class_count[0][0])

    return np.array(predict_result)


def knn(x_train, y_train, x_test, window_size, k, train_abstr_labels=None, test_abstr_labels=None, weights=None):
    """
    kNN classifier
    :param x_train: training set
    :param y_train: training labels
    :param x_test: testing set
    :param window_size: the warping window size of DTW
    :param k: the number of nearest neighbors
    :param train_abstr_labels: cluster labels of training abstractions
    :param test_abstr_labels: cluster labels of testing abstractions
    :param weights: WED weights of clusters
    :return: predicted testing labels
    """
    predict_result = []
    train_set_size = len(x_train)
    test_set_size = len(x_test)
    distances = np.zeros(train_set_size)

    # compute cDTW/APDTW
    for i in range(test_set_size):
        for j in range(train_set_size):
            if weights is None:
                distances[j], path = cdtw(x_test[i], x_train[j], window_size)
            else:
                distances[j], path = cdtw(x_test[i], x_train[j], window_size,
                                          test_abstr_labels[i], train_abstr_labels[j], weights)

        sorted_dist = np.argsort(distances)
        class_count = {}

        # count the classes of k nearest neighbors
        for x in range(k):
            sort_label = y_train[sorted_dist[x]]
            class_count[sort_label] = class_count.get(sort_label, 0) + 1

        # get the predicted label from the sorted class number
        sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
        predict_result.append(sorted_class_count[0][0])

    return np.array(predict_result)


def score(predict_label, label_truth):
    """
    compute the classification accuracy
    """
    count = 0

    for i in range(len(predict_label)):
        if predict_label[i] == label_truth[i]:
            count += 1

    accuracy = count / len(predict_label)

    return accuracy


def adapdtw_test(train_trans, train_labels, test_trans, test_labels, train_abstr_labels,
                 test_abstr, wed_weight, cluster_center, window_size):
    """
    test the classification accuracy on testing set
    :param train_trans: abstraction sequence of training set
    :param train_labels: class labels of training set
    :param test_trans: abstraction sequence of testing set
    :param test_labels: class labels of testing set
    :param train_abstr_labels: cluster labels of training abstractions
    :param test_abstr: testing abstractions
    :param wed_weight: WED weights of clusters
    :param cluster_center: cluster centers
    :param window_size: the size of warping window
    :return: classification accuracy
    """

    # get the inter-cluster weights
    weight_k = np.zeros(cluster_center.shape)
    for i in range(weight_k.shape[0]):
        weight_k[i] = wed_weight[i][i]

    # get the cluster labels of testing abstractions
    cluster_labels = partition_abstr(test_abstr, weight_k, cluster_center)
    test_abstr_labels = reshape(test_trans, cluster_labels)

    # classify test set
    start = time.time()
    predict_labels = knn(train_trans, train_labels, test_trans, window_size, 1,
                         train_abstr_labels, test_abstr_labels, wed_weight)
    end = time.time()
    runtime = round(end - start, 2)
    accuracy = score(predict_labels, test_labels)

    return accuracy, runtime


def reshape(data, labels):
    """
    Reshape labels by the form of data
    """

    abstr_labels = []
    st = 0

    for i in range(len(data)):
        abstr_labels.append(labels[st:st + data[i].shape[0]])
        st += data[i].shape[0]

    return abstr_labels
