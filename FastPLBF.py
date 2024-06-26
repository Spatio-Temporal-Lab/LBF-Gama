from utils.ThresMaxDivDP import MaxDivDP, ThresMaxDiv
from utils.OptimalFPR import OptimalFPR
from utils.SpaceUsed import SpaceUsed
from utils.const import INF
from PLBF import PLBF

import time
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


class FastPLBF(PLBF):
    def __init__(self, pos_keys: list, pos_scores: list[float], neg_scores: list[float], F: float, N: int, k: int, bf_size: float):
        """
        Args:
            pos_keys (list): keys
            pos_scores (list[float]): scores of keys
            neg_scores (list[float]): scores of non-keys
            F (float): target overall fpr
            N (int): number of segments
            k (int): number of regions
        """

        # assert
        assert (isinstance(pos_keys, list))
        assert (isinstance(pos_scores, list))
        assert (len(pos_keys) == len(pos_scores))
        assert (isinstance(neg_scores, list))
        assert (isinstance(F, float))
        assert (0 < F < 1)
        assert (isinstance(N, int))
        assert (isinstance(k, int))

        for score in pos_scores:
            assert (0 <= score <= 1)
        for score in neg_scores:
            assert (0 <= score <= 1)

        self.F = F
        self.N = N
        self.k = k
        self.n = len(pos_keys)

        segment_thre_list, g, h = self.divide_into_segments(pos_scores, neg_scores)
        self.find_best_t_and_f(segment_thre_list, g, h, bf_size)
        self.insert_keys(pos_keys, pos_scores)

    def find_best_t_and_f(self, segment_thre_list, g, h, bf_size):
        minSpaceUsed = bf_size
        t_best = None
        f_best = None

        DPKL, DPPre = MaxDivDP(g, h, self.N, self.k)
        for j in range(self.k, self.N + 1):
            t = ThresMaxDiv(DPPre, j, self.k, segment_thre_list)
            if t is None:
                continue
            f = OptimalFPR(g, h, t, self.F, self.k)
            if minSpaceUsed > SpaceUsed(g, h, t, f, self.n):
                minSpaceUsed = SpaceUsed(g, h, t, f, self.n)
                t_best = t
                f_best = f

        self.t = t_best
        self.f = f_best
        self.memory_usage_of_backup_bf = minSpaceUsed


def run(path,query_path, F, N, k, bf_size):
    data = pd.read_csv(path)
    query_data = pd.read_csv(query_path)
    negative_sample = data.loc[(data['label'] == 0)]
    positive_sample = data.loc[(data['label'] == 1)]
    train_negative = negative_sample.sample(frac=0.8)
    query_negative = query_data.loc[(query_data['label'] == 0)]


    pos_keys = list(positive_sample['key'])
    pos_scores = list(positive_sample['score'])
    train_neg_scores = list(train_negative['score'])

    query_neg_keys = list(query_negative['key'])
    query_neg_scores = list(query_negative['score'])

    plbf = FastPLBF(pos_keys, pos_scores, train_neg_scores, F, N, k, bf_size)

    # test
    fp_cnt = 0
    total = len(query_neg_keys)
    for key, score in zip(query_neg_keys, query_neg_scores):
        if plbf.contains(key, score):
            fp_cnt += 1
    print(f"fpr: {float(fp_cnt) / total}")
    return float(fp_cnt) / total




