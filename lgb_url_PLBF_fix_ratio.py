import copy
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

import lib.lgb_url
from plbf.FastPLBF_M import FastPLBF_M

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

positive_items = np.concatenate((X_train[y_train == 1], X_test[y_test == 1]), axis=0)
negative_items = np.concatenate((X_train[y_train == 0], X_test[y_test == 0]), axis=0)
positive_samples = X_sample[y_sample == 1]
negative_samples = X_sample[y_sample == 0]

positive_train_urls = df_train[df_train['url_type'] == 1]['url']
positive_test_urls = df_test[df_test['url_type'] == 1]['url']
positive_urls = pd.concat([positive_train_urls, positive_test_urls])
positive_urls_list = positive_urls.tolist()

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

for ratio in np.arange(0.2, 1.0, 0.1):
    fpr_sum = 0
    for size in range(64 * 1024, 320 * 1024 + 1, 64 * 1024):
        epoch_max = 200
        best_fpr = 1.0
        best_plbf = None
        best_epoch = 0
        bst = lgb.Booster(params=params, train_set=train_data)

        start_time = time.perf_counter_ns()
        for i in range(epoch_max):
            bst.update(train_data)
            if lib.lgb_url.lgb_get_model_size(bst) >= size * ratio:
                break
            # bf_bytes = size - lib.lgb_url.lgb_get_model_size(bst)
            # if bf_bytes <= 0:
            #     break
            # if i < 1:
            #     continue
            #
            # pos_scores = bst.predict(positive_samples).tolist()
            # neg_scores = bst.predict(negative_samples).tolist()
            # plbf = FastPLBF_M(positive_urls_list, pos_scores, neg_scores, bf_bytes * 8.0, 50, 5)
            # cur_fpr = plbf.get_fpr()
            # if cur_fpr < best_fpr:
            #     best_plbf = copy.deepcopy(plbf)
            #     best_fpr = cur_fpr
            #     best_epoch = i + 1

        # best_bst = lgb.Booster(model_str=bst.model_to_string(num_iteration=best_epoch))

        model_size = lib.lgb_url.lgb_get_model_size(bst)
        print("模型在内存中所占用的大小（字节）:", model_size)
        print(f"best epoch:", best_epoch)

        bf_bytes = size - model_size

        fp_cnt = 0
        query_negative = X_query
        query_neg_keys = query_urls
        query_neg_scores = bst.predict(X_query)
        total = len(query_negative)

        pos_scores = bst.predict(positive_items).tolist()
        neg_scores = bst.predict(negative_items).tolist()
        plbf = FastPLBF_M(positive_urls_list, pos_scores, neg_scores, bf_bytes * 8.0, 50, 5)
        plbf.insert_keys(positive_urls_list, pos_scores)
        end_time = time.perf_counter_ns()
        print(f'use {(end_time - start_time) / 1000000}ms')

        for key, score in zip(query_neg_keys, query_neg_scores):
            if plbf.contains(key, score):
                fp_cnt += 1
        fpr = float(fp_cnt) / total
        fpr_sum += fpr
        print(f"fpr: {fpr}")
    print(fpr_sum)
