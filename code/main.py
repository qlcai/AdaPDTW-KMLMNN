import math

import os
import time

import numpy as np
import pandas as pd

import kmlmnn
from transform import pf_transform, cca_transform, dsa_transform, paa_transform
from getdata import read_uts_data
from ewkm import get_ewkm
from knn import knn, adapdtw_test, reshape, score, knn_dist


def para_assess():
    """
    Experiments on parameter assessment
    """
    for f in ['SmallKitchenAppliances', 'ScreenType', 'HandOutlines', 'Earthquakes']:
        print(f)

        writer = pd.ExcelWriter('../result/para_assess/' + f + '.xlsx')

        train_data, train_labels, test_data, test_labels = read_uts_data('../data/' + f)
        train_trans, train_vectors = pf_transform(train_data, seg_num=10)
        test_trans, test_vectors = pf_transform(test_data, seg_num=10)

        w_acc = []      # assess cluster number w
        for cluster_num in [4, 8, 16, 32, 64]:
            acc = []
            for count in range(10):
                cluster_labels, cluster_centers, wed_weights = get_ewkm(train_vectors, cluster_num, lamda=0.5)
                train_vectors_labels = reshape(train_trans, cluster_labels)

                alg = kmlmnn.JoinLearn(train_trans, train_labels, train_vectors_labels,
                                       wed_weights, cluster_centers, near_num=3)
                loss, weight_iter, clus_cent_iter, train_vectors_label_iter = alg.opt(iter_num=10)
                accuracy = adapdtw_test(train_trans, train_labels, test_trans, test_labels, train_vectors_label_iter[-1],
                                        test_vectors, weight_iter[-1], clus_cent_iter[-1], None)
                acc.append(accuracy)

            accuracy = sum(acc) / len(acc)
            w_acc.append([cluster_num, accuracy])
            print([f, 'cluster num:%d' % cluster_num, accuracy])

        pd.DataFrame(w_acc).to_excel(writer, sheet_name='cluster_num')

        sigma_acc = []      # assess coefficient of PFLMNN
        for sigma in [0.01, 0.1, 1, 10, 100]:
            acc = []
            for count in range(10):
                cluster_labels, cluster_centers, wed_weights = get_ewkm(train_vectors, 4, lamda=0.5)
                train_vectors_labels = reshape(train_trans, cluster_labels)

                alg = kmlmnn.JoinLearn(train_trans, train_labels, train_vectors_labels,
                                       wed_weights, cluster_centers, near_num=3, sigma=sigma)
                loss, weight_iter, clus_cent_iter, train_vectors_label_iter = alg.opt(iter_num=10)
                accuracy = adapdtw_test(train_trans, train_labels, test_trans, test_labels,
                                        train_vectors_label_iter[-1],
                                        test_vectors, weight_iter[-1], clus_cent_iter[-1], None)
                acc.append(accuracy)

            accuracy = sum(acc) / len(acc)
            sigma_acc.append([sigma, accuracy])
            print([f, 'sigma:%f' % sigma, accuracy])

        pd.DataFrame(sigma_acc).to_excel(writer, sheet_name='sigma')

        lamda_acc = []      # assess coefficient of weight entropy
        for lamda in [0.01, 0.1, 1, 10, 100]:
            acc = []
            for count in range(10):
                cluster_labels, cluster_centers, wed_weights = get_ewkm(train_vectors, 4, lamda=0.5)
                train_vectors_labels = reshape(train_trans, cluster_labels)

                alg = kmlmnn.JoinLearn(train_trans, train_labels, train_vectors_labels,
                                       wed_weights, cluster_centers, near_num=3, lamda=lamda)
                loss, weight_iter, clus_cent_iter, train_vectors_label_iter = alg.opt_comp(iter_num=10)
                accuracy = adapdtw_test(train_trans, train_labels, test_trans, test_labels,
                                        train_vectors_label_iter[-1], test_vectors, weight_iter[-1],
                                        clus_cent_iter[-1], None)
                acc.append(accuracy)

            accuracy = sum(acc) / len(acc)
            lamda_acc.append([lamda, accuracy])
            print([f, 'lamda:%f' % lamda, accuracy])

        pd.DataFrame(lamda_acc).to_excel(writer, sheet_name='lamda')

        writer.save()


def convergence():
    """
    Experiments on convergence
    """
    writer = pd.ExcelWriter('../result/convergence.xlsx')
    val_err_result = {}

    for f in ['SmallKitchenAppliances', 'ScreenType', 'HandOutlines', 'Earthquakes']:
        print(f)

        train_data, train_labels, test_data, test_labels = read_uts_data('../data/' + f)

        # get validation set
        vl = math.floor(len(train_data) / 5)
        valid_data = train_data[:vl]
        valid_labels = train_labels[:vl]
        train_val_data = np.delete(train_data, list(range(0, vl)), axis=0)
        train_val_labels = np.delete(train_labels, list(range(0, vl)), axis=0)

        train_val_trans, train_val_vectors = pf_transform(train_val_data, seg_num=10)
        valid_trans, valid_vectors = pf_transform(valid_data, seg_num=10)

        cluster_labels, cluster_centers, wed_weights = get_ewkm(train_val_vectors, cluster_num=4, lamda=0.5)
        train_vectors_labels = reshape(train_val_trans, cluster_labels)

        alg = kmlmnn.JoinLearn(train_val_trans, train_val_labels, train_vectors_labels,
                               wed_weights, cluster_centers, near_num=3)

        valid_err, weight_iter, clus_cent_iter, train_vectors_label_iter = alg.opt_valid(train_val_trans,
                                                                                         train_val_labels,
                                                                                         valid_trans, valid_labels,
                                                                                         valid_vectors, iter_num=20)
        val_err_result[f] = valid_err

    pd.DataFrame(val_err_result).to_excel(writer, sheet_name='validation_error')
    writer.save()


def dtw():
    """
    Experiments with DTW
    """
    writer = pd.ExcelWriter('../result/DTW.xlsx')
    dataset = pd.read_excel('../DataSummary.xlsx')
    result = []

    for f in os.listdir('../data/'):
        if f not in dataset['Name'].values:
            continue

        train_data, train_labels, test_data, test_labels = read_uts_data('../data/' + f)

        # classify test set
        start = time.time()

        predict_labels = knn_dist(train_data, train_labels, test_data, 1, 'dtw')

        end = time.time()
        runtime = round(end - start, 2)

        accuracy = score(predict_labels, test_labels)

        result.append([f, accuracy, runtime])
        print(result)

    pd.DataFrame(result).to_excel(writer)
    writer.save()


def cdtw():
    """
    Experiments with cDTW
    """
    writer = pd.ExcelWriter('../result/cDTW.xlsx')
    dataset = pd.read_excel('../DataSummary.xlsx')

    result = []
    for f in os.listdir('../data/'):
        if f not in dataset['Name'].values:
            continue

        train_data, train_labels, test_data, test_labels = read_uts_data('../data/' + f)

        w = dataset[dataset['Name'] == f]['Window'].iloc[0]

        # classify testing set
        start = time.time()

        predict_labels = knn_dist(train_data, train_labels, test_data, 1, 'dtw', w)

        end = time.time()
        runtime = round(end - start, 2)

        test_accuracy = score(predict_labels, test_labels)
        result.append([f, test_accuracy, runtime])

    pd.DataFrame(result).to_excel(writer)
    writer.save()


def pdtw():
    """
    Experiments with PDTW
    """
    writer = pd.ExcelWriter('../result/PDTW.xlsx')
    dataset = pd.read_excel('../DataSummary.xlsx')
    result = []

    for f in os.listdir('../data/'):
        if f not in dataset['Name'].values:
            continue

        train_data, train_labels, test_data, test_labels = read_uts_data('../data/' + f)

        # five-fold cross validation
        vl = math.floor(len(train_data) / 5)
        para = {}

        for seg in range(2, 21):
            print([f, 'segmentation number:%d' % seg])
            acc = []
            for i in range(5):
                valid_data = train_data[vl * i:vl * (i + 1)]
                valid_labels = train_labels[vl * i:vl * (i + 1)]
                train_val_data = np.delete(train_data, list(range(vl * i, vl * (i + 1))), axis=0)
                train_val_labels = np.delete(train_labels, list(range(vl * i, vl * (i + 1))), axis=0)

                train_val_trans = paa_transform(train_val_data, seg_num=seg)
                valid_trans = paa_transform(valid_data, seg_num=seg)

                predict_labels = knn(train_val_trans, train_val_labels, valid_trans, None, 1)
                acc.append(score(predict_labels, valid_labels))

            para[seg] = sum(acc) / len(acc)

        seg = max(para, key=para.get)

        # PAA transformation
        train_trans = paa_transform(train_data, seg)
        test_trans = paa_transform(test_data, seg)

        # classify test set
        start = time.time()

        predict_labels = knn(train_trans, train_labels, test_trans, None, 1)

        end = time.time()
        runtime = round(end - start, 2)

        accuracy = score(predict_labels, test_labels)
        result.append([f, seg, accuracy, runtime])

    pd.DataFrame(result).to_excel(writer)
    writer.save()


def pfdtw():
    """
    Experiments with PFDTW
    """
    writer = pd.ExcelWriter('../result/PFDTW.xlsx')
    dataset = pd.read_excel('../DataSummary.xlsx')
    result = []

    for f in os.listdir('../data/'):
        if f not in dataset['Name'].values:
            continue

        train_data, train_labels, test_data, test_labels = read_uts_data('../data/' + f)

        # five-fold cross validation
        vl = math.floor(len(train_data) / 5)
        para = {}
        for seg in range(2, 21):
            print([f, 'segmentation number:%d' % seg])
            acc = []
            for i in range(5):
                valid_data = train_data[vl * i:vl * (i + 1)]
                valid_labels = train_labels[vl * i:vl * (i + 1)]
                train_val_data = np.delete(train_data, list(range(vl * i, vl * (i + 1))), axis=0)
                train_val_labels = np.delete(train_labels, list(range(vl * i, vl * (i + 1))), axis=0)

                train_val_trans, train_val_vectors = pf_transform(train_val_data, seg_num=seg)
                valid_trans, valid_vectors = pf_transform(valid_data, seg_num=seg)

                predict_labels = knn(train_val_trans, train_val_labels, valid_trans, None, 1)
                acc.append(score(predict_labels, valid_labels))

            para[seg] = sum(acc) / len(acc)

        seg = max(para, key=para.get)

        # PF transformation
        train_trans, train_vectors = pf_transform(train_data, seg)
        test_trans, test_vectors = pf_transform(test_data, seg)

        # classify test set
        start = time.time()

        predict_labels = knn(train_trans, train_labels, test_trans, None, 1)

        end = time.time()
        runtime = round(end - start, 2)

        accuracy = score(predict_labels, test_labels)
        result.append([f, seg, accuracy, runtime])

    pd.DataFrame(result).to_excel(writer)
    writer.save()


def threeddtw():
    """
    Experiments with 3-D DTW
    """
    writer = pd.ExcelWriter('../result/3DDTW.xlsx')
    dataset = pd.read_excel('../DataSummary.xlsx')
    result = []

    for f in os.listdir('../data/'):
        if f not in dataset['Name'].values:
            continue

        train_data, train_labels, test_data, test_labels = read_uts_data('../data/' + f)

        train_trans, train_vectors, seg1 = cca_transform(train_data, K=3, s=0.6)
        test_trans, test_vectors, seg2 = cca_transform(test_data, K=3, s=0.6)
        avg_seg = (seg1 + seg2) / 2

        # classify test set
        start = time.time()

        predict_labels = knn(train_trans, train_labels, test_trans, None, 1)

        end = time.time()
        runtime = round(end - start, 2)

        accuracy = score(predict_labels, test_labels)

        result.append([f, avg_seg, accuracy, runtime])

    pd.DataFrame(result).to_excel(writer)
    writer.save()


def dsadtw():
    """
    Experiments with DSADTW
    """
    writer = pd.ExcelWriter('../result/DSADTW.xlsx')
    dataset = pd.read_excel('../DataSummary.xlsx')
    result = []

    for f in os.listdir('../data/'):
        if f not in dataset['Name'].values:
            continue

        train_data, train_labels, test_data, test_labels = read_uts_data('../data/' + f)
        train_data = pd.DataFrame(train_data)
        test_data = pd.DataFrame(test_data)

        # compute difference
        deriv_train_data = pd.concat([train_data[1] - train_data[0],
                                      pd.DataFrame(train_data.iloc[:, 2:].values - train_data.iloc[:, :-2].values) / 2,
                                      train_data.iloc[:, -1] - train_data.iloc[:, -2]], axis=1)
        deriv_test_data = pd.concat([test_data[1] - test_data[0],
                                     pd.DataFrame(test_data.iloc[:, 2:].values - test_data.iloc[:, :-2].values) / 2,
                                     test_data.iloc[:, -1] - test_data.iloc[:, -2]], axis=1)

        train_trans, seg1 = dsa_transform(deriv_train_data.values)
        test_trans, seg2 = dsa_transform(deriv_test_data.values)
        avg_seg = (seg1 + seg2) / 2

        # classify test set
        start = time.time()

        predict_labels = knn(train_trans, train_labels, test_trans, None, 1)

        end = time.time()
        runtime = round(end - start, 2)

        accuracy = score(predict_labels, test_labels)

        result.append([f, avg_seg, accuracy, runtime])

    pd.DataFrame(result).to_excel(writer)
    writer.save()


def pfadapdtw():
    """
    Experiments on PF-AdaPDTW
    """
    writer = pd.ExcelWriter('../result/accuracy.xlsx', sheet_name='PFAdaPDTW')
    dataset = pd.read_excel('../DataSummary.xlsx')
    result = []

    for f in dataset['Name'].values:
        train_data, train_labels, test_data, test_labels = read_uts_data('../data/' + f)  # read raw data

        # five-fold cross validation for learning super parameters
        vl = math.floor(len(train_data) / 5)    # valid set size
        para = {}                               # store the accuracy for each (seg, cluster_num)
        for seg in range(2, 20):                # segment number
            for cluster_num in range(2, 10):    # abstraction cluster number
                print(['Training:   ', f, '    segment num:%d' % seg, '    cluster num:%d' % cluster_num])
                acc = []                        # store the accuracy of each fold
                for i in range(5):
                    valid_data = train_data[vl * i:vl * (i + 1)]        # validate set
                    valid_labels = train_labels[vl * i:vl * (i + 1)]    # validate labels
                    train_val_data = np.delete(train_data, list(range(vl * i, vl * (i + 1))), axis=0)       # training set
                    train_val_labels = np.delete(train_labels, list(range(vl * i, vl * (i + 1))), axis=0)   # training set

                    # PAR transformation with PF features
                    train_val_trans, train_val_abstrs = pf_transform(train_val_data, seg_num=seg)
                    valid_trans, valid_abstrs = pf_transform(valid_data, seg_num=seg)

                    # cluster all training abstracts by EWKM
                    cluster_labels, cluster_centers, wed_weights = get_ewkm(train_val_abstrs, cluster_num, lamda=0.5)
                    train_abstr_labels = reshape(train_val_trans, cluster_labels)

                    # DTW-KMLMNN learning process with iter_num iterations
                    alg = kmlmnn.JoinLearn(train_val_trans, train_val_labels, train_abstr_labels,
                                           wed_weights, cluster_centers, near_num=3)
                    weight_iter, clus_cent_iter, train_abstrs_label_iter = alg.opt(iter_num=10)

                    # validation accuracy
                    acc.append(adapdtw_test(train_val_trans, train_val_labels, valid_trans, valid_labels,
                                            train_abstrs_label_iter[-1], valid_abstrs, weight_iter[-1],
                                            clus_cent_iter[-1], None))

                para[(seg, cluster_num)] = sum(acc) / len(acc)

        seg, cluster_num = max(para, key=para.get)      # get optimal parameters

        train_trans, train_abstrs = pf_transform(train_data, seg_num=seg)
        test_trans, test_abstrs = pf_transform(test_data, seg_num=seg)

        print(['Testing:    ', f, ' segment num:%d' % seg, '    cluster num:%d' % cluster_num])

        # train model for classification
        cluster_labels, cluster_centers, wed_weights = get_ewkm(train_abstrs, cluster_num, lamda=0.5)
        train_abstr_labels = reshape(train_trans, cluster_labels)

        alg = kmlmnn.JoinLearn(train_trans, train_labels, train_abstr_labels,
                               wed_weights, cluster_centers, near_num=3)
        weight_iter, clus_cent_iter, train_abstr_label_iter = alg.opt(iter_num=10)

        # classify testing set
        start = time.time()

        accuracy = adapdtw_test(train_trans, train_labels, test_trans, test_labels, train_abstr_label_iter[-1],
                                test_abstrs, weight_iter[-1], clus_cent_iter[-1], None)
        end = time.time()
        runtime = round(end - start, 2)

        result.append([f, seg, cluster_num, accuracy, runtime])

        print(["Finish classification: ", f])

    pd.DataFrame(result).to_excel(writer)
    writer.save()


if __name__ == "__main__":
    para_assess()
    convergence()
    dtw()
    cdtw()
    pdtw()
    pfdtw()
    threeddtw()
    dsadtw()
    pfadapdtw()
