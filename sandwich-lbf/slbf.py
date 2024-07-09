import math

import numpy as np
import pandas as pd
from Bloom_filter import BloomFilter


class SLBF:
    def __init__(self, keys, filter_size_b1, filter_size_b2, threshold):
        '''
        keys: df in the following form
            (index)     data    label    score
        '''
        self.filter_size_b1 = filter_size_b1
        self.filter_size_b2 = filter_size_b2
        self.threshold = threshold

        self.initial_keys = keys
        if filter_size_b1 > 0:
            self.initial_bf = BloomFilter(len(self.initial_keys),
                                          filter_size_b1 * len(self.initial_keys))
            self.initial_bf.insert(self.initial_keys.iloc[:, 0])
        else:
            self.initial_bf = None
        self.backup_keys = keys[(keys.iloc[:, -1] <= threshold)]
        self.backup_bf = BloomFilter(len(self.backup_keys), filter_size_b2 * len(self.initial_keys))
        self.backup_bf.insert(self.backup_keys.iloc[:, 0])

    def query(self, query_set):
        ml_false_positive = (query_set.iloc[:,
                             -1] > self.threshold)  # maschera falsi positivi generati dal modello rispetto alla soglia considerata,
        ml_true_negative = (query_set.iloc[:,
                            -1] <= self.threshold)  # maschera veri negativi generati dal modello rispetto alla soglia considerata
        # Calcolo FPR
        initial_bf_false_positive = self.initial_bf.test(query_set.iloc[:, 0],
                                                         single_key=False) if self.initial_bf is not None else np.full(
            len(query_set), True)  # if initial BF is not present, all of query samples are "false positive"
        ml_false_positive_list = query_set.iloc[:, 0][(initial_bf_false_positive) & (ml_false_positive)]
        ml_true_negative_list = query_set.iloc[:, 0][(initial_bf_false_positive) & (ml_true_negative)]
        backup_bf_false_positive = self.backup_bf.test(ml_true_negative_list, single_key=False)
        total_false_positive = sum(backup_bf_false_positive) + len(ml_false_positive_list)

        return total_false_positive

    def query_single_key(self, key, score):
        if self.initial_bf.test(key):
            if score > self.threshold:
                return True
            else:
                if self.backup_bf.test(key):
                    return True
                else:
                    return False
        else:
            return False


def train_slbf(filter_size, query_train_set, keys):
    # train_dataset = np.array(pd.concat([query_train_set, keys]).iloc[:, -1])
    # thresholds_list = [np.quantile(train_dataset, i * (1 / quantile_order)) for i in
    #                    range(1, quantile_order)] if quantile_order < len(train_dataset) else np.sort(train_dataset)
    # thresh_third_quart_idx = (3 * len(thresholds_list) - 1) // 4
    thresholds_list = [round(i * 0.01, 2) for i in range(1, 100)]

    fp_opt = query_train_set.shape[0]
    slbf_opt = None  # cambiare
    #    print("thresholds_list:", thresholds_list)
    for threshold in thresholds_list:
        ml_false_positive = (query_train_set.iloc[:, -1] > threshold)
        ml_false_negative = (keys.iloc[:, -1] <= threshold)

        FP = query_train_set[ml_false_positive].iloc[:, 0].size / query_train_set.iloc[:, 0].size
        FN = keys[ml_false_negative].iloc[:, 0].size / keys.iloc[:, 0].size

        if FP == 0.0:
            print("FP = 0, skip")
            # filter_opt = learned_bloom_filter.main(classifier_score_path, correct_size_filter, other)
            slbf_opt = SLBF(keys, 0, filter_size, threshold)
            continue
        if FN == 1.0 or FN == 0.0:
            # print("FP is equal to 1.0, or FN is equal to 0 or 1, skipping threshold")
            continue
        if FP + FN > 1:  # If FP + FN >= 1, the budget b2 becomes negative
            # print("FP + FN >= 1, skipping threshold")
            continue

        b2 = FN * math.log(FP / ((1 - FP) * ((1 / FN) - 1)), 0.6185)
        b1 = filter_size - b2
        if b1 <= 0:  # Non serve avere SLBF
            print("b1 = 0")
            b1 = 0
            break

        # print(f"FP: {FP}, FN: {FN}, b: {filter_size}, b1: {b1}, b2: {b2}")

        slbf = SLBF(keys, b1, b2, threshold)
        fp_items = slbf.query(query_train_set)
        #        print(f"\tFalse positive items: {fp_items}")
        if fp_items < fp_opt:
            fp_opt = fp_items
            slbf_opt = slbf
        #            print(f"False positive items: {fp_items} - Current threshold: {threshold}")
        if slbf_opt is None:
            print("FN + FP >= 1 with all the thresold, is impossible to build a SLBF")
    fp_items = slbf_opt.query(query_train_set)
    print(f"Chosen thresholds: {slbf_opt.threshold} - False positive items: {fp_items}")

    return slbf_opt, fp_opt


def get_slbf_opt(positive_sample, train_negative, R_sum):
    b = R_sum / len(positive_sample)
    slbf_opt, fp_opt = train_slbf(b, train_negative, positive_sample)
    return slbf_opt


def run(R_sum, path, model, X_query, y_query, query_urls):
    data = pd.read_csv(path)
    negative_sample = data.loc[(data['label'] == 0)]
    positive_sample = data.loc[(data['label'] == 1)]
    # train_negative = negative_sample.sample(frac=0.8)
    train_negative = negative_sample
    print("start running")
    slbf_opt = get_slbf_opt(positive_sample, train_negative, R_sum)
    fn = 0
    fp = 0
    cnt_ml = 0
    cnt_bf = 0
    total = len(X_query)
    print(f"query count = {total}")
    prediction_results = model.predict(X_query)

    for i in range(total):
        true_label = y_query[i]
        url = query_urls[i]
        score = prediction_results[i]
        result = slbf_opt.query_single_key(key=url, score=score)
        if true_label == 0 and result == 1:
            fp += 1
        elif true_label == 1 and result == 0:
            fn += 1

    print(f"fp: {fp}")
    print(f"total: {total}")
    print(f"fpr: {float(fp) / total}")
    print(f"fnr: {float(fn) / total}")
    print(f"cnt_ml: {cnt_ml}")
    print(f"cnt_bf: {cnt_bf}")
    return float(fp) / total

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_path', action="store", dest="data_path", type=str, required=True,
#                         help="path of the dataset")
#     parser.add_argument('--size_of_Sandwiched', action="store", dest="R_sum", type=int, required=True,
#                         help="size of the Ada-BF")
#     result = parser.parse_known_args()
#     main(result[0].data_path, result[0].R_sum, result[1])
