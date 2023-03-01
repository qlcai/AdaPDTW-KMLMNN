import math

import numpy as np


def cdtw(x, y, w=None, x_lab=None, y_lab=None, weights=None):
    """
    compute the APDTW and the owp under warping window
    :param x: a time series sample
    :param y: a time series sample
    :param x_lab: the cluster labels for each abstraction of x
    :param y_lab: the cluster labels for each abstraction of y
    :param w: the size of warping window
    :param weights: the WED weights of the abstraction clusters
    :return: the APDTW and the owp between x and y
    """
    if w is None:
        w1 = len(x)
        w2 = len(y)
    else:
        w1, w2, wl = math.ceil(max(len(x), len(y)) * w)

    D = np.zeros((len(x), len(y))) + np.inf

    if weights is None:
        D[0, 0] = pointwise_dist(x[0], y[0])
        for i in range(1, w2):
            D[0, i] = D[0, i - 1] + pointwise_dist(x[0], y[i])
        for i in range(1, w1):
            D[i, 0] = D[i - 1, 0] + pointwise_dist(x[i], y[0])
    else:
        D[0, 0] = pointwise_dist(x[0], y[0], weights[x_lab[0]][y_lab[0]])
        for i in range(1, w2):
            D[0, i] = D[0, i - 1] + pointwise_dist(x[0], y[i], weights[x_lab[0]][y_lab[i]])
        for i in range(1, w1):
            D[i, 0] = D[i - 1, 0] + pointwise_dist(x[i], y[0], weights[x_lab[i]][y_lab[0]])

    for i in range(1, len(x)):
        if w is None:
            st = 1
            ed = len(y)
        elif i - wl <= 0:
            st = 1
            ed = i + wl + 1
            if ed > len(y):
                ed = len(y)
        elif i + wl >= len(y):
            st = i - wl
            ed = len(y)
        else:
            st = i - wl
            ed = i + wl + 1
        for j in range(st, ed):
            if weights is None:
                wei = None
            else:
                wei = weights[x_lab[i]][y_lab[j]]

            D[i, j] = pointwise_dist(x[i], y[j], wei) + min(D[i - 1, j], D[i - 1, j - 1], D[i, j - 1])

    path = _traceback(D)

    return D[-1, -1], path


def pointwise_dist(x, y, weight=None):
    """
    compute the square difference between abstractions
    :param x: an abstraction
    :param y: an abstraction
    :param weight: wed weight
    :return: a real number
    """
    dist = np.power(x - y, 2)
    if weight is not None:
        dist = np.multiply(weight, dist)
    weight_dist_sum = np.sum(dist)

    return weight_dist_sum


def _traceback(d):
    """
    trace the owp of DTW
    :param d:the pointwise distance matrix
    :return: two index arrays
    """
    i, j = np.array(d.shape) - 1
    p, q = [i], [j]

    while (i > 0) or (j > 0):
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            n = np.argmin([d[i-1, j-1], d[i-1, j], d[i, j-1]])
            if n == 0:
                i -= 1
                j -= 1
            elif n == 1:
                i -= 1
            else:
                j -= 1

        p.insert(0, i)
        q.insert(0, j)

    return np.array(p), np.array(q)
