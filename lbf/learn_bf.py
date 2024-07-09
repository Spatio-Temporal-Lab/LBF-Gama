import numpy as np
import pandas as pd

from bf import BloomFilter


def Find_Optimal_Parameters(max_thres, min_thres, R_sum, train_negative, positive_sample):
    FP_opt = train_negative.shape[0]

    for threshold in np.arange(min_thres, max_thres + 10 ** (-6), 0.01):
        url = positive_sample.loc[(positive_sample['score'] <= threshold), 'url']
        n = len(url)
        bloom_filter = BloomFilter(n, R_sum)
        bloom_filter.insert(url)
        ML_positive = train_negative.loc[(train_negative['score'] > threshold), 'url']
        bloom_negative = train_negative.loc[(train_negative['score'] <= threshold), 'url']
        BF_positive = bloom_filter.test(bloom_negative, single_key=False)
        FP_items = sum(BF_positive) + len(ML_positive)
        print('Threshold: %f, False positive items: %d' % (round(threshold, 2), FP_items))
        if FP_opt > FP_items:
            FP_opt = FP_items
            thres_opt = threshold
            bloom_filter_opt = bloom_filter

    return bloom_filter_opt, thres_opt


def run(R_sum, path, model, X_query, y_query, query_urls):
    data = pd.read_csv(path)
    negative_sample = data.loc[(data['label'] == 0)]
    positive_sample = data.loc[(data['label'] == 1)]
    # train_negative = negative_sample.sample(frac=0.8)
    train_negative = negative_sample

    bloom_filter_opt, thresholds_opt = Find_Optimal_Parameters(0.99, 0.01, R_sum, train_negative, positive_sample)
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
        if score > thresholds_opt:
            if true_label == 0:
                fp += 1
                cnt_ml += 1
        else:
            if bloom_filter_opt.test(url) == 1 and true_label == 0:
                fp += 1
                cnt_bf += 1
            elif bloom_filter_opt.test(url) == 0 and true_label == 1:
                fn = fn + 1

    print(f"fp: {fp}")
    print(f"total: {total}")
    print(f"fpr: {float(fp) / total}")
    print(f"fnr: {float(fn) / total}")
    print(f"cnt_ml: {cnt_ml}")
    print(f"cnt_bf: {cnt_bf}")
    return float(fp) / total

# '''
# Implement learned Bloom filter
# '''
# if __name__ == '__main__':
#     '''Stage 1: Find the hyper-parameters (spare 30% samples to find the parameters)'''
#     bloom_filter_opt, thres_opt = Find_Optimal_Parameters(max_thres, min_thres, R_sum, train_negative, positive_sample)
#
#     '''Stage 2: Run Ada-BF on all the samples'''
#     ### Test URLs
#     ML_positive = negative_sample.loc[(negative_sample['score'] > thres_opt), 'url']
#     bloom_negative = negative_sample.loc[(negative_sample['score'] <= thres_opt), 'url']
#     score_negative = negative_sample.loc[(negative_sample['score'] < thres_opt), 'score']
#     BF_positive = bloom_filter_opt.test(bloom_negative, single_key = False)
#     FP_items = sum(BF_positive) + len(ML_positive)
#     print('False positive items: %d' % FP_items)
