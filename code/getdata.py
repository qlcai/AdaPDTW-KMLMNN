import numpy as np
import os

import pandas as pd


def read_uts_data(path):
    """
    read the raw data
    :param path: the path of data file
    :return: datasets
    """

    df = os.listdir(path)
    if len(df) == 3:
        df.remove('README.md')

    if 'TEST' in df[0]:
        test_path = path + '/' + df[0]
        train_path = path + '/' + df[1]
    else:
        test_path = path + '/' + df[1]
        train_path = path + '/' + df[0]

    # read training set
    train_data = pd.read_csv(train_path, header=None, sep='\t')
    train_data = train_data.reindex(np.random.permutation(train_data.index)).reset_index(drop=True)
    train_labels = train_data[0]
    train_data.drop([0], axis=1, inplace=True)

    # read test set
    test_data = pd.read_csv(test_path, header=None, sep='\t')
    test_labels = test_data[0]
    test_data.drop([0], axis=1, inplace=True)

    # preprocess with centered moving average of window 7
    train_data = train_data.T
    train_data.iloc[:7, :] = train_data.iloc[:7, :].mean().values
    for i in range(7, train_data.shape[0]):
        train_data.iloc[i, :] = train_data.iloc[i - 3:i + 4, :].mean().values
    train_data.iloc[-7:, :] = train_data.iloc[-7:, :].mean().values
    train_data = train_data.T

    test_data = test_data.T
    test_data.iloc[:7, :] = test_data.iloc[:7, :].mean().values
    for i in range(7, test_data.shape[0]):
        test_data.iloc[i, :] = test_data.iloc[i - 3:i + 4, :].mean().values
    test_data.iloc[-7:, :] = test_data.iloc[-7:, :].mean().values
    test_data = test_data.T

    return train_data.values, train_labels.values, test_data.values, test_labels.values


def z_normalize(data):
    """
    z-score normalization
    """

    avg = data.mean(axis=1)
    std = data.std(axis=1)
    data = data.sub(avg, axis=0).div(std, axis=0).values

    return data
