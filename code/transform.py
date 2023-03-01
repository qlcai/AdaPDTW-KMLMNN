import numpy as np

import pandas as pd
import math


def paa_transform(data, seg_num):
    """
    compute the permutation entropy
    :param data: time series dataset
    :param seg_num: segment number
    :return paa: PAA sequence
    """
    n, m = data.shape
    paa = []

    for i in range(n):
        ts = data[i]
        ts_paa = []
        al = math.ceil(len(ts) / seg_num)
        for j in range(1, seg_num):
            if j == seg_num - 1:
                sub = ts[j * al:]
            else:
                sub = ts[j * al:(j + 1) * al]
            ts_paa.append(np.average(sub))

        paa.append(ts_paa)

    return paa


def pf_transform(data, seg_num):
    """
    PAR transformation with the polynomial fitting features
    :param data: time series dataset
    :param seg_num: segment number
    :return psa: PF sequence
    """
    n, m = data.shape
    pf = []
    abstracts = pd.DataFrame()

    for i in range(n):
        ts = data[i]
        ts_pf = []
        al = math.ceil(len(ts) / seg_num)
        for j in range(seg_num):
            if j == seg_num - 1:
                sub = ts[j * al:]
                al = len(sub)
            else:
                sub = ts[j * al:(j + 1) * al]
            if al < 6:
                continue
            mean = sum(sub) / al                                            # PF coefficients of zero order
            res1 = np.polyfit(x=np.arange(al), y=sub, deg=1)                # PF coefficients of 1st order
            res2 = np.polyfit(x=np.arange(al), y=sub, deg=2)                # PF coefficients of 2nd order
            res3 = np.polyfit(x=np.arange(al), y=sub, deg=3)                # PF coefficients of 3rd order
            feature = [res1[0]] + res2[:2].tolist() + res3[:3].tolist()     # PF abstraction
            ts_pf.append([mean] + feature)

        # z-normalize all features
        ts_pf = pd.DataFrame(ts_pf)
        for j in range(ts_pf.shape[1]):
            ts_pf[j] = (ts_pf[j] - ts_pf[j].mean()) / ts_pf[j].std()
        ts_pf.fillna(0, inplace=True)
        pf.append(ts_pf.values)
        abstracts = abstracts.append(ts_pf, ignore_index=True)

    return pf, abstracts.values


def cca_transform(data, K, s):
    """
    compute the cca representation
    :param data: z-normalized time series dataset
    :param K: control limit
    :param s: threshold distance parameter
    :return cca: sequence of 3d-tuple
    """
    n, m = data.shape
    cca = []
    vectors = pd.DataFrame()
    avg_seg = 0

    for i in range(n):
        ts = data[i]
        ts_cca = []
        st = 0
        pre_intval = get_interval(ts[0], K, s)

        # discretize each value and segment
        ts_len = len(ts)
        for j in range(1, ts_len):
            intval = get_interval(ts[j], K, s)

            if (j > 0 and intval != pre_intval) or j == ts_len:
                ed = j
                v = sum(ts[st:ed]) / (ed - st)
                ts_cca.append([v, ed - st])
                st = ed
                pre_intval = intval

        avg_seg += len(ts_cca)

        # normalize the duration feature
        if not ts_cca:
            ts_cca.append([1, 1])
        ts_cca = pd.DataFrame(ts_cca)
        ts_cca[1] = (ts_cca[1] - ts_cca[1].mean()) / ts_cca[1].std()
        ts_cca.fillna(0, inplace=True)
        cca.append(ts_cca.values)
        vectors = vectors.append(ts_cca, ignore_index=True)

    avg_seg /= n

    return cca, vectors.values, avg_seg


def get_interval(x, K, s):
    """
    get the interval number for each data
    """
    if x + K < 0:
        return 0
    elif x + K > 2 * K:
        return 2 * K / s + 1
    else:
        return math.ceil((x + K) / s)


def dsa_transform(data):
    """
    compute the DSA representation
    :param data: z-normalized time series dataset
    :return dsa: DSA sequence
    """
    n, m = data.shape
    dsa = []
    avg_seg = 0

    for i in range(n):
        ts = data[i]
        m = len(ts)
        ts_dsa = []
        st = 1
        for j in range(1, m):
            if np.abs(np.average(ts[st:j])-ts[j]) > np.std(ts):
                ts_dsa.append(np.arctan(np.average(ts[st:j])))
                st = j
        if m - st > 2:
            ts_dsa.append(np.arctan(np.average(ts[st:m])))

        avg_seg += len(ts_dsa)
        dsa.append(ts_dsa)

    avg_seg /= n

    return dsa, avg_seg
