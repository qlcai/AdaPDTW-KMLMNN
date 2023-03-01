import numpy as np
import pandas as pd
import copy
from apdtw import wdtw, pointwise_dist
from knn import apdtw_test


class JoinLearn:

    def __init__(self, train_data, train_label, train_abstr_label, weight,
                 cluster_centers, near_num, window_size=None, sigma=None, lamda=None):
        self.train_data = train_data  # training data
        self.train_label = train_label  # training class labels
        self.train_abstr_label = train_abstr_label  # abstract cluster labels of traning set
        self.weight = weight  # WED weights
        self.cluster_center = cluster_centers  # abstract cluster centers
        self.cluster_num = weight.shape[0]  # abstract cluster number
        self.near_num = near_num  # nearest neighbor number of PFLMNN
        self.window_size = window_size  # warping window size of DTW
        self.train_num = len(train_data)  # training set size
        self.dtw_dist_mat = np.zeros((self.train_num, self.train_num))  # pointwise cost matrix of DTW
        self.path_list = [0] * len(train_data)  # all OWPs between training samples
        self.target_set = np.array(list(np.arange(self.train_num)) * self.near_num).reshape(
            self.near_num, self.train_num).T  # all target neighbors to training samples
        self.imposter_set = np.arange(self.train_num)  # all imposters to training samples
        self.update_apdtw()
        self.update_target_imposter()
        self.sigma = self.get_ewkm_loss() / self.get_lmnn_loss()  # coefficient of PFLMNN in loss function
        self.lamda = abs(
            self.get_ewkm_loss() / self.get_entropy_loss())  # coefficient of weight entropy in loss function

        if sigma:
            self.sigma *= sigma
        if lamda:
            self.lamda *= lamda

    def update_apdtw(self):
        """
        Update the APDTW and owp between each pair of time series
        """
        print("update_apdtw")
        for i in range(self.train_num):
            tmp_path_list = []
            for j in range(self.train_num):
                if j < i:
                    path = (self.path_list[j][i][1], self.path_list[j][i][0])
                    self.dtw_dist_mat[i, j] = self.dtw_dist_mat[j, i]
                    tmp_path_list.append(path)
                else:
                    # abstract cluster labels
                    x_labels = self.train_abstr_label[i]
                    y_labels = self.train_abstr_label[j]

                    # compute AdaPDTW and OWP
                    dist, path = wdtw(self.train_data[i], self.train_data[j],
                                      self.window_size, x_labels, y_labels, self.weight)
                    self.dtw_dist_mat[i, j] = dist
                    tmp_path_list.append(path)

            self.path_list[i] = tmp_path_list

    def update_target_imposter(self):
        """
        Update the target and imposter sets for each sample
        """
        print("update_target_imposter")
        train_label = np.array(self.train_label)
        sorted_dist_mat = np.argsort(self.dtw_dist_mat)
        temp = np.reshape(sorted_dist_mat, (1, -1))
        labels = pd.DataFrame(np.reshape(train_label[temp], sorted_dist_mat.shape))
        indx = (labels == pd.DataFrame(labels[:][0]).values)
        indx.drop(columns=0, inplace=True)
        for i in range(indx.shape[0]):
            x = indx.iloc[i, :]
            ind = x[x].index.tolist()
            L = len(ind)
            if L >= self.near_num:
                self.target_set[i] = np.array(ind[:self.near_num])
            else:
                self.target_set[i, :L] = np.array(ind)
            try:
                n = list(x).index(False)
            except ValueError:
                continue
            else:
                self.imposter_set[i] = n + 1

    def update_weight(self):
        """
        Update WED weights by Eq.(18) and (20)
        """
        print("update_weight")
        theta = np.zeros((self.cluster_num, self.cluster_num, self.train_data[0].shape[1]))
        d = np.zeros((self.cluster_num, self.train_data[0].shape[1]))

        # compute theta_pq by Eq.(15)
        for i in range(self.cluster_num):
            for j in range(self.cluster_num):
                theta[i][j] = self.get_theta_pq(i, j)

        # compute D_k by Eq.(21)
        for i in range(self.cluster_num):
            d[i] = self.get_d_k(i)

        # compute weights
        for i in range(self.cluster_num):
            for j in range(self.cluster_num):

                # compute w_k by Eq.(20)
                if i == j:
                    numerator = np.exp(np.divide(-self.sigma * theta[i][j] - d[i], self.lamda))
                    denominator = np.sum(numerator)
                    if denominator != 0:
                        self.weight[i][j] = np.divide(numerator, denominator)

                # compute w_pq by Eq.(18)
                else:
                    numerator = np.exp(np.divide(-self.sigma * theta[i][j], self.lamda))
                    denominator = np.sum(numerator)
                    if denominator != 0:
                        self.weight[i][j] = np.divide(numerator, denominator)

    def get_theta_pq(self, p, q):
        """
        Compute the inter parameter theta_pq by Eq.(15)
        """
        theta_pq = np.zeros(self.train_data[0].shape[1])
        for i in range(self.train_num):
            lm = self.imposter_set[i]
            tmp = np.zeros(theta_pq.shape[0])
            path_2 = self.path_list[i][lm]
            sub_2 = np.zeros(theta_pq.shape[0])

            # the 2nd term within the square bracket of Eq.(15)
            for k in range(path_2[0].shape[0]):
                re_index = (path_2[0][k], path_2[1][k])
                if self.train_abstr_label[i][re_index[0]] != p or self.train_abstr_label[lm][re_index[1]] != q:
                    continue
                sub_2 += np.power(self.train_data[i][re_index[0]] - self.train_data[lm][re_index[1]], 2)

            for j in range(self.target_set[i].shape[0]):
                z_index = self.target_set[i][j]
                if 1 - self.dtw_dist_mat[i][lm] + self.dtw_dist_mat[i][z_index] <= 0:
                    continue
                path_1 = self.path_list[i][z_index]
                sub_1 = np.zeros(theta_pq.shape[0])

                # the 1st term within the square bracket of Eq.(15)
                for k in range(path_1[1].shape[0]):
                    re_index = (path_1[0][k], path_1[1][k])
                    if self.train_abstr_label[i][re_index[0]] != p or self.train_abstr_label[z_index][re_index[1]] != q:
                        continue
                    sub_1 += np.power(self.train_data[i][re_index[0]] - self.train_data[z_index][re_index[1]], 2)
                tmp += (sub_1 - sub_2)
            theta_pq += tmp

        return theta_pq

    def get_d_k(self, k):
        """
        Compute D_k by Eq.(21)
        """
        d_k = np.zeros(self.train_data[0].shape[1])
        for i in range(self.train_num):
            for j in range(self.train_data[i].shape[0]):
                if self.train_abstr_label[i][j] == k:
                    d_k += np.power(self.cluster_center[k] - self.train_data[i][j], 2)
        return d_k

    def update_abstr_label_1(self):
        """
        Update abstract cluster labels
        """
        print("update_abstr_label")
        weight_k = np.zeros((self.cluster_num, self.train_data[0].shape[1]))
        for i in range(self.cluster_num):
            weight_k[i] = self.weight[i][i]

        for i in range(self.train_num):
            for j in range(self.train_data[i].shape[0]):
                dist = np.power((self.cluster_center - self.train_data[i][j]), 2)
                weight_dist = np.multiply(dist, weight_k)
                weight_dist_sum = np.sum(weight_dist, axis=1)
                self.train_abstr_label[i][j] = np.argmin(weight_dist_sum)

    def update_abstr_label(self):
        """
        Update cluster labels for each abstraction by Eq.(23)
        """
        print("update_abstr_label")
        weight_k = np.zeros((self.cluster_num, self.train_data[0].shape[1]))

        for i in range(self.cluster_num):
            weight_k[i] = self.weight[i][i]

        for i in range(self.train_num):
            for j in range(self.train_data[i].shape[0]):
                dist = np.power((self.cluster_center - self.train_data[i][j]), 2)
                weight_dist = np.multiply(dist, weight_k)
                weight_dist_sum = np.sum(weight_dist, axis=1)
                lmnn = np.zeros(self.cluster_num)

                sub_2 = np.zeros(self.cluster_num)
                lm = self.imposter_set[i]
                path_2 = self.path_list[i][lm]

                for n in range(path_2[0].shape[0]):
                    re_index = (path_2[0][n], path_2[1][n])
                    c_2 = self.train_abstr_label[lm][re_index[1]]
                    if re_index[0] == j:
                        for k in range(self.cluster_num):
                            c_1 = k
                            weight_dist = pointwise_dist(self.train_data[i][re_index[0]],
                                                         self.train_data[lm][re_index[1]],
                                                         self.weight[c_1][c_2])
                            sub_2[k] += weight_dist
                    else:
                        c_1 = self.train_abstr_label[i][re_index[0]]
                        weight_dist = pointwise_dist(self.train_data[i][re_index[0]],
                                                     self.train_data[lm][re_index[1]],
                                                     self.weight[c_1][c_2])
                        sub_2 += weight_dist

                for m in range(self.near_num):
                    sub_1 = np.zeros(self.cluster_num)
                    z_index = self.target_set[i][m]
                    path_1 = self.path_list[i][z_index]

                    for x in range(path_1[1].shape[0]):
                        re_index = (path_1[0][x], path_1[1][x])
                        c_2 = self.train_abstr_label[z_index][re_index[1]]

                        if re_index[0] == j:
                            for k in range(self.cluster_num):
                                c_1 = k
                                weight_dist = pointwise_dist(self.train_data[i][re_index[0]],
                                                             self.train_data[z_index][re_index[1]],
                                                             self.weight[c_1][c_2])
                                sub_1[k] += weight_dist
                        else:
                            c_1 = self.train_abstr_label[i][re_index[0]]
                            weight_dist = pointwise_dist(self.train_data[i][re_index[0]],
                                                         self.train_data[z_index][re_index[1]],
                                                         self.weight[c_1][c_2])
                            sub_1 += weight_dist

                    temp = 1 - sub_2 + sub_1
                    temp[temp < 0] = 0
                    lmnn += temp

                dist_list = weight_dist_sum + self.sigma * lmnn
                self.train_abstr_label[i][j] = np.argmin(dist_list)

    def update_cluster_center(self):
        """
        Update cluster centers by Eq.(24)
        """
        print("update_cluster_center")

        tmp = np.zeros((self.cluster_num, self.train_data[0].shape[1]))
        cnt = np.zeros((self.cluster_num, 1))

        for i in range(self.train_num):
            for j in range(self.train_data[i].shape[0]):
                k = self.train_abstr_label[i][j]
                tmp[k] += self.train_data[i][j]
                cnt[k] += 1

        ind = np.where(cnt != 0)[0]
        self.cluster_center[ind] = np.divide(tmp[ind], cnt[ind])

    def loss_function(self):
        """
        Compute the loss function by Eq.(9)
        """
        print("loss_function")

        ewkm_loss = self.get_ewkm_loss()  # compute the loss of EWKM
        lmnn_loss = self.get_lmnn_loss()  # compute the loss of PFLMNN
        entropy_loss = self.get_entropy_loss()  # compute the loss of weight entropy
        loss = ewkm_loss + self.sigma * lmnn_loss + self.lamda * entropy_loss  # compute the total loss

        return loss

    def get_ewkm_loss(self):
        """
        Compute the loss of EWKM by Eq.(9)
        """
        ewkm_loss = 0

        for i in range(self.train_num):
            for j in range(self.train_data[i].shape[0]):
                for k in range(self.cluster_num):
                    if self.train_abstr_label[i][j] == k:
                        weight_dist = pointwise_dist(self.cluster_center[k],
                                                     self.train_data[i][j],
                                                     self.weight[k][k])
                        ewkm_loss += weight_dist

        return ewkm_loss

    def get_lmnn_loss(self):
        """
        Compute the loss of PFLMNN by Eq.(9)
        """
        lmnn_loss = 0

        for i in range(self.train_num):
            sub_2 = 0
            lm = self.imposter_set[i]
            path_2 = self.path_list[i][lm]

            # compute the 2nd term of E(W)
            for k in range(path_2[0].shape[0]):
                re_index = (path_2[0][k], path_2[1][k])
                c_1 = self.train_abstr_label[i][re_index[0]]
                c_2 = self.train_abstr_label[lm][re_index[1]]
                weight_dist = pointwise_dist(self.train_data[i][re_index[0]],
                                             self.train_data[lm][re_index[1]],
                                             self.weight[c_1][c_2])
                sub_2 += weight_dist

            # compute the 1st term of E(W)
            for j in range(self.target_set[i].shape[0]):
                z_index = self.target_set[i][j]
                path_1 = self.path_list[i][z_index]
                sub_1 = 0
                for k in range(path_1[1].shape[0]):
                    re_index = (path_1[0][k], path_1[1][k])
                    c_1 = self.train_abstr_label[i][re_index[0]]
                    c_2 = self.train_abstr_label[z_index][re_index[1]]
                    weight_dist = pointwise_dist(self.train_data[i][re_index[0]],
                                                 self.train_data[z_index][re_index[1]],
                                                 self.weight[c_1][c_2])
                    sub_1 += weight_dist

                # hinge loss
                if 1 - sub_2 + sub_1 > 0:
                    lmnn_loss += 1 - sub_2 + sub_1

        return lmnn_loss

    def get_entropy_loss(self):
        """
        Compute the loss of weight entropy by Eq.(9)
        """
        entropy_loss = 0

        for i in range(self.cluster_num):
            for j in range(self.cluster_num):
                entropy_loss += np.sum(np.nan_to_num(np.multiply(
                    self.weight[i][j], np.log(self.weight[i][j]))))

        return entropy_loss

    def opt(self, iter_num):
        """
        Optimize DTW-KMLMNN
        :param iter_num: iteration number
        :return:weight_iter: the WED weight list of each iteration
                cluster_center_iter: the cluster centers list of each iteration
                train_abstr_label_iter: the abstract cluster labels of each iteration
        """
        weight_iter = []
        cluster_center_iter = []
        train_abstr_label_iter = []

        for i in range(iter_num):
            print("optimization iteration: %d" % i)

            if i > 0:
                self.update_apdtw()  # update the owp between samples
                self.update_target_imposter()  # update target and imposter sets
            self.update_weight()  # update w_k and w_pq
            self.update_abstr_label_1()  # update cluster labels of abstractions
            self.update_cluster_center()  # update cluster centers

            weight_iter.append(copy.deepcopy(self.weight))
            cluster_center_iter.append(copy.deepcopy(self.cluster_center))
            train_abstr_label_iter.append(copy.deepcopy(self.train_abstr_label))

        return weight_iter, cluster_center_iter, train_abstr_label_iter

    def opt_valid(self, train_val_trans, train_val_labels, valid_trans, valid_labels, valid_vectors, iter_num):
        """
        Optimize DTW-KMLMNN
        :param iter_num: iteration number
        :return:valid_err: validate error at each iteration
                weight_iter: the WED weight list of each iteration
                cluster_center_iter: the cluster centers list of each iteration
                train_abstr_label_iter: the abstract cluster labels of each iteration
        """
        valid_err = []
        weight_iter = []
        cluster_center_iter = []
        train_abstr_label_iter = []

        for i in range(iter_num):
            print("optimization iteration: %d" % i)

            if i > 0:
                self.update_apdtw()  # update the owp between samples
                self.update_target_imposter()  # update target and imposter sets
            self.update_weight()  # update w_k and w_pq
            self.update_abstr_label_1()  # update cluster labels of abstractions
            self.update_cluster_center()  # update cluster centers

            weight_iter.append(copy.deepcopy(self.weight))
            cluster_center_iter.append(copy.deepcopy(self.cluster_center))
            train_abstr_label_iter.append(copy.deepcopy(self.train_abstr_label))

            acc = apdtw_test(train_val_trans, train_val_labels, valid_trans, valid_labels,
                             self.train_abstr_label, valid_vectors, self.weight, self.cluster_center, None)
            valid_err.append(1 - acc)

        return valid_err, weight_iter, cluster_center_iter, train_abstr_label_iter
