import numpy as np
import random


def get_ewkm(data, cluster_num, lamda):
    """
    Cluster all abstractions by EWKM
    :param data: abstract set
    :param cluster_num: integer
    :param lamda: real number
    :return: cluster_labels: cluster labels that abstracts belong to,
             cluster_centers: cluster centers
             wed_weights: WED weights
    """
    iteration = 10
    loss = np.zeros(iteration)

    # initialize weights
    cluster_weights = np.zeros((cluster_num, data.shape[1])) + np.divide(1.0, data.shape[1])

    # initialize cluster centers
    n = data.shape[0]
    rand_index = np.array(random.sample(range(n), cluster_num))
    cluster_centers = data[rand_index]

    cluster_labels = None

    # update cluster_labels, cluster_centers, and cluster_weights with 10 iterations
    for i in range(iteration):
        cluster_labels = partition_abstr(data, cluster_weights, cluster_centers)
        cluster_centers = compute_cluster_center(data, cluster_labels, cluster_num, cluster_centers)
        cluster_weights = compute_weight(data, cluster_weights, cluster_centers, cluster_labels, lamda)
        loss[i] = loss_function(data, cluster_centers, cluster_labels, cluster_weights, lamda)

    print(loss)

    # initialize weights of WEDs within and between clusters
    wed_weights = np.zeros((cluster_num, cluster_num, data.shape[1])) + np.divide(1.0, data.shape[1])

    # store the inter-cluster weights
    for i in range(cluster_num):
        wed_weights[i, i] = cluster_weights[i]

    return cluster_labels, cluster_centers, wed_weights


def partition_abstr(data, cluster_weights, cluster_centers):
    """
    Assign cluster label for each abstract
    """
    n = data.shape[0]
    cluster_labels = np.empty(n, dtype=int)

    for i in range(n):
        dist = np.power((cluster_centers - data[i]), 2)
        weight_dist = np.multiply(dist, cluster_weights)
        weight_dist_sum = np.sum(weight_dist, axis=1)
        cluster_labels[i] = np.argmin(weight_dist_sum)

    return cluster_labels


def compute_cluster_center(data, cluster_labels, cluster_num, cluster_centers):
    """
    Update cluster centers
    """
    for k in range(cluster_num):
        index = np.where(cluster_labels == k)[0]

        if not index.any():
            continue

        temp = data[index]
        all_dimension_sum = np.sum(temp, axis=0)
        cluster_centers[k] = all_dimension_sum / index.size

    return cluster_centers


def compute_weight(data, cluster_weights, cluster_centers, cluster_labels, lamda):
    """
    Update cluster weights
    """
    cluster_num = cluster_centers.shape[0]
    distance_sum = np.zeros((cluster_num, data.shape[1]), dtype=float)

    for k in range(cluster_num):
        index = np.where(cluster_labels == k)[0]
        temp = data[index]
        distance = np.power((temp - cluster_centers[k]), 2)
        distance_sum[k] = np.sum(distance, axis=0)

    for k in range(cluster_num):
        numerator = np.exp(np.divide(-distance_sum[k], lamda))
        denominator = np.sum(numerator)
        if denominator != 0:
            cluster_weights[k] = np.divide(numerator, denominator)

    return cluster_weights


def loss_function(data, cluster_centers, cluster_labels, cluster_weights, lamda):
    """
    Compute the loss function
    """
    loss = 0
    cluster_num = cluster_centers.shape[0]

    for k in range(cluster_num):
        index = np.where(cluster_labels == k)[0]
        temp = data[index]
        distance = np.power((temp - cluster_centers[k]), 2)
        weight_distance = np.multiply(distance, cluster_weights[k])
        temp = np.nan_to_num(np.multiply(cluster_weights[k], np.log(cluster_weights[k])))
        temp = lamda * np.sum(temp)
        loss = loss + np.sum(weight_distance) + temp

    return loss
