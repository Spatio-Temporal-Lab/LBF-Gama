import time

import lightgbm as lgb
import numpy as np
import pandas as pd

import lib.bf_util
import lib.lgb_url

df_train = pd.read_csv('dataset/url_train.csv')
df_test = pd.read_csv('dataset/url_test.csv')
df_query = pd.read_csv('dataset/url_query.csv')

train_urls = df_train['url']
test_urls = df_test['url']
query_urls = df_query['url']

X_train = df_train.drop(columns=['url', 'url_type']).values.astype(np.float32)
y_train = df_train['url_type'].values.astype(np.float32)
X_test = df_test.drop(columns=['url', 'url_type']).values.astype(np.float32)
y_test = df_test['url_type'].values.astype(np.float32)
X_query = df_query.drop(columns=['url', 'url_type']).values.astype(np.float32)
y_query = df_query['url_type'].values.astype(np.float32)

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
        fpr_bf = lib.bf_util.get_fpr(bf_count, bf_bytes)
        fpr_lgb = fp / n_false
        fpr_lbf = fpr_lgb + (1 - fpr_lgb) * fpr_bf

        if fpr_lbf < best_fpr_lbf:
            best_thresh = thresh
            best_fpr_lbf = fpr_lbf

    # print(f'best thresh = {best_thresh} and best fpr = {best_fpr_lbf}')
    return best_thresh, best_fpr_lbf


for ratio in np.arange(0.1, 1.0, 100):
    fpr_sum = 0
    for size in range(64 * 1024, 320 * 1024 + 1, 64 * 1024):
        start_time = time.perf_counter_ns()
        bst = None
        best_bst = None
        best_fpr = 1.0
        best_threshold = 0.5
        epoch_max = 200
        best_epoch = 0
        bst = lgb.Booster(params=params, train_set=train_data)

        for i in range(epoch_max):
            bst.update(train_data)
            if lib.lgb_url.lgb_get_model_size(bst) >= size * ratio:
                break

        bf_bytes = size - lib.lgb_url.lgb_get_model_size(bst)

        # 对训练集进行预测
        train_pred = bst.predict(X_train)
        test_pred = bst.predict(X_test)

        # 拼接预测结果
        all_predictions = np.concatenate([train_pred, test_pred])
        all_true_labels = np.concatenate([y_train, y_test])
        best_thresh, best_fpr_lbf = evaluate_thresholds(all_predictions, all_true_labels, bf_bytes)

        # 保存最佳模型
        if best_bst is None or best_fpr_lbf < best_fpr:
            best_bst = bst.__copy__()
            best_threshold = best_thresh
            best_fpr = best_fpr_lbf

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
        fpr_sum += fpr
    print(f"fpr_sum =  {fpr_sum:.2f}")
