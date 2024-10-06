import numpy as np
import pandas as pd
from sklearn.utils import murmurhash3_32
import random
import serialize
import argparse
from pathlib import Path
import time


class hashfunc(object):
    def __init__(self, m):
        self.m = m
        self.ss = random.randint(1, 99999999)

    def __call__(self, x):
        x=str(x)
        return murmurhash3_32(x, seed=self.ss) % self.m


'''
Class for Standard Bloom filter
'''


class BloomFilter():
    def __init__(self, n, hash_len):
        self.n = n
        self.hash_len = int(hash_len)
        if (self.hash_len == 0):
            raise SyntaxError('The hash table is empty')
        if (self.n > 0) & (self.hash_len > 0):
            self.k = max(1, int(self.hash_len / n * 0.6931472))
        elif (self.n == 0):
            self.k = 1
        self.h = []
        for i in range(self.k):
            self.h.append(hashfunc(self.hash_len))
        self.table = np.zeros(self.hash_len, dtype=int)

    def insert(self, key):
        if self.hash_len == 0:
            raise SyntaxError('cannot insert to an empty hash table')
        for i in key:
            for j in range(self.k):
                t = self.h[j](i)
                self.table[t] = 1

    # def test(self, key):
    #     test_result = 0
    #     match = 0
    #     if self.hash_len > 0:
    #         for j in range(self.k):
    #             t = self.h[j](key)
    #             match += 1*(self.table[t] == 1)
    #         if match == self.k:
    #             test_result = 1
    #     return test_result

    def test(self, keys, single_key=True):
        if single_key:
            test_result = 0
            match = 0
            if self.hash_len > 0:
                for j in range(self.k):
                    t = self.h[j](keys)
                    match += 1 * (self.table[t] == 1)
                if match == self.k:
                    test_result = 1
        else:
            test_result = np.zeros(len(keys))
            ss = 0
            if self.hash_len > 0:
                for key in keys:
                    match = 0
                    for j in range(self.k):
                        t = self.h[j](key)
                        match += 1 * (self.table[t] == 1)
                    if match == self.k:
                        test_result[ss] = 1
                    ss += 1
        return test_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action="store", dest="data_path", type=str, required=True,
                        help="path of the dataset")
    parser.add_argument('--size_of_BF', action="store", dest="R_sum", type=int, required=True, help="size of the BF")
    parser.add_argument('--pos_ratio', action="store", dest="pos_ratio", type=float, required=True,
                        help="size of the BF", default=0.7)
    parser.add_argument('--neg_ratio', action="store", dest="neg_ratio", type=float, required=True,
                        help="size of the BF", default=0.7)
    parser.add_argument("--negTest_ratio", action="store", dest="negTest_ratio", type=float, default=1.0)
    parser.add_argument("--test_path", action="store", dest="test_path", type=str, default=None)
    parser.add_argument("--save_path", action="store", dest="save_path", type=str, default=None)
    seed = 22012022
    rs = np.random.RandomState(seed)
    random.seed(seed)

    args = parser.parse_args()
    data_path = Path(args.data_path)
    data_test_path = Path(args.test_path) if args.test_path is not None else None
    R_sum = args.R_sum
    pos_ratio = args.pos_ratio
    neg_ratio = args.neg_ratio
    negTest_ratio = args.negTest_ratio
    data_test_path = args.test_path
    dataset = serialize.load_dataset(data_path)
    neg_label = serialize.find_neg_label(dataset)
    dataset_test = serialize.load_dataset(data_path) if data_test_path is not None else None
    print(
        f"Total samples: {len(dataset.index)}. (Pos, Neg): ({len(dataset[(dataset['label'] == 1)])}, {len(dataset[(dataset['label'] == neg_label)])})")
    data, query_negative = serialize.divide_dataset(dataset, dataset_test, pos_ratio, neg_ratio, negTest_ratio, rs)
    del (dataset)
    print(
        f"Samples for filters training: {len(data.index)}. (Pos, Neg): ({len(data[(data['label'] == 1)])}, {len(data[(data['label'] == neg_label)])})")
    print(f"Samples for filters testing: {len(query_negative.index)}")
    #    print(query_negative.iloc[:, 0].head())

    negative_sample = data.loc[(data.iloc[:, -1] == neg_label)]  # label?
    positive_sample = data.loc[(data.iloc[:, -1] == 1)]
    query = positive_sample.iloc[:, 0]
    n = len(query)
    bloom_filter = BloomFilter(n, R_sum)
    bloom_filter.insert(query)
    start = time.time()
    n1 = bloom_filter.test(query_negative.iloc[:, 0], single_key=False)
    end = time.time()
    print('False positive rate: ', sum(n1) / len(query_negative), ", Time:", (end - start) / len(query_negative))
    if args.save_path:
        file1 = open(args.save_path, 'a')
        file1.write(
            str(data_path) + "," + str(R_sum) + "," + str(np.round(sum(n1) / len(query_negative), 5)) + "," + str(
                (end - start) / len(query_negative)) + "\n")