import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization

import lib.bf_util
import lib.lgb_url

positive_frac = 0.1
df_train = pd.read_csv('dataset/url_train.csv')
df_test = pd.read_csv('dataset/url_test.csv')
df_query = pd.read_csv('dataset/url_query.csv')
df_sample = pd.read_csv('dataset/url_sample' + str(positive_frac) + '.csv')

train_urls = df_train['url']
test_urls = df_test['url']
query_urls = df_query['url']

X_train = df_train.drop(columns=['url', 'url_type']).values.astype(np.float32)
y_train = df_train['url_type'].values.astype(np.float32)
X_test = df_test.drop(columns=['url', 'url_type']).values.astype(np.float32)
y_test = df_test['url_type'].values.astype(np.float32)
X_query = df_query.drop(columns=['url', 'url_type']).values.astype(np.float32)
y_query = df_query['url_type'].values.astype(np.float32)
X_sample = df_sample.drop(columns=['url', 'url_type']).values.astype(np.float32)
y_sample = df_sample['url_type'].values.astype(np.float32)

train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
test_data = lgb.Dataset(X_test, label=y_test, free_raw_data=False)

# 设置参数
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'verbose': -1
}

n_true = df_train[df_train['url_type'] == 1].shape[0] + df_test[df_test['url_type'] == 1].shape[0]
n_false = df_train[df_train['url_type'] == 0].shape[0] + df_test[df_test['url_type'] == 0].shape[0]
n_test = len(df_test)


def evaluate_thresholds(prediction_results, y_true, bf_bytes):
    sorted_indices = np.argsort(prediction_results)
    sorted_predictions = prediction_results[sorted_indices]
    sorted_true = y_true[sorted_indices]

    fp = n_false
    tp = 0
    best_thresh = 0
    best_fpr_lbf = 1.0

    unique_sorted_predictions, idx = np.unique(sorted_predictions, return_index=True)

    n = len(unique_sorted_predictions)
    for i in range(n):
        thresh = unique_sorted_predictions[i]

        if i < n - 1:
            count_1 = np.sum(sorted_true[idx[i]:idx[i + 1]])
            tp += count_1
            fp -= idx[i + 1] - idx[i] - count_1
        else:
            count_1 = np.sum(sorted_true[idx[i]:n])
            tp += count_1
            fp -= n - idx[i] - count_1

        bf_count = tp
        fpr_bf = lib.bf_util.get_fpr(bf_count / positive_frac, bf_bytes)
        fpr_lgb = fp / n_false
        fpr_lbf = fpr_lgb + (1 - fpr_lgb) * fpr_bf

        if fpr_lbf < best_fpr_lbf:
            best_thresh = thresh
            best_fpr_lbf = fpr_lbf

    # print(f'best thresh = {best_thresh} and best fpr = {best_fpr_lbf}')
    return best_thresh, best_fpr_lbf


for size in range(64 * 1024, 320 * 1024 + 1, 64 * 1024):
    epoch_max = 200
    epoch_each = 1
    bst = None
    model_sizes = []
    best_fpr = 1.0
    best_threshold = 0.5
    best_epoch = 0
    cur_epoch = 0

    # Train the model up to `epoch_max` rounds, recording model size at each round

    start_time = time.perf_counter_ns()
    best_score = float('inf')
    fpr_map = dict()

    bst = lgb.Booster(params=params, train_set=train_data)


    # Define the objective function for Bayesian optimization
    def objective(query_epoch):
        global cur_epoch, epoch_max
        query_epoch = int(query_epoch)
        if query_epoch > epoch_max:
            return -999

        if fpr_map.__contains__(query_epoch):
            return fpr_map[query_epoch]

        while cur_epoch < query_epoch:
            cur_epoch += 1
            bst.update(train_data)
            size_temp = lib.lgb_url.lgb_get_model_size(bst)
            if size_temp > size:
                epoch_max = cur_epoch - 1
                return -999
            model_sizes.append(size_temp)

        model_size = model_sizes[query_epoch - 1]  # Model size at `num_iteration`
        bf_bytes = size - model_size

        sample_pred = bst.predict(X_sample, num_iteration=query_epoch)
        best_thresh, best_fpr_lbf = evaluate_thresholds(sample_pred, y_sample, bf_bytes)

        global best_fpr, best_threshold, best_epoch
        if best_fpr_lbf < best_fpr:
            best_threshold = best_thresh
            best_fpr = best_fpr_lbf
            best_epoch = query_epoch

        fpr_map[query_epoch] = best_fpr_lbf

        return -best_fpr_lbf


    # 使用贝叶斯优化来寻找最佳 epoch
    def optimize_epochs():
        pbounds = {'query_epoch': (1, 200)}  # 设置 epoch 范围
        optimizer = BayesianOptimization(
            f=objective,
            pbounds=pbounds,
            verbose=0,
            random_state=42,
        )

        optimizer.maximize(init_points=5, n_iter=20)


    optimize_epochs()
    # best_bst = lgb.train(params, train_data, num_boost_round=int(best_epoch), valid_sets=[test_data])
    best_bst = lgb.Booster(model_str=bst.model_to_string(num_iteration=best_epoch))

    model_size = lib.lgb_url.lgb_get_model_size(best_bst)

    data_negative = lib.lgb_url.lgb_validate_url(best_bst, X_train, y_train, train_urls, X_test, y_test, test_urls,
                                                 best_threshold)
    bloom_size = size - model_size

    bloom_filter = lib.lgb_url.create_bloom_filter(dataset=data_negative, bf_size=bloom_size)
    end_time = time.perf_counter_ns()
    print(f'use {(end_time - start_time) / 1000000}ms')

    # 访问布隆过滤器的 num_bits 属性
    num_bits = bloom_filter.num_bits

    # 将比特位转换为字节（8 bits = 1 byte）
    memory_in_bytes = num_bits / 8

    fpr = lib.lgb_url.lgb_query_url(best_bst, bloom_filter, X_query, y_query, query_urls, best_threshold, False)
