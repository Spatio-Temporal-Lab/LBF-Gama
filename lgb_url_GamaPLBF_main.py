import copy
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

import lib.lgb_url
from plbf.FastPLBF_M import FastPLBF_M

df_train = pd.read_csv('dataset/url_train.csv')
df_test = pd.read_csv('dataset/url_test.csv')
df_query = pd.read_csv('dataset/url_query.csv')

# 筛选出正类样本的URLs
positive_train_urls = df_train[df_train['url_type'] == 1]['url']
positive_test_urls = df_test[df_test['url_type'] == 1]['url']

# 合并正类样本的URLs
positive_urls = pd.concat([positive_train_urls, positive_test_urls])

# 转换成list
positive_urls_list = positive_urls.tolist()

query_urls = df_query['url']

X_train = df_train.drop(columns=['url', 'url_type']).values.astype(np.float32)
y_train = df_train['url_type'].values.astype(np.float32)
X_test = df_test.drop(columns=['url', 'url_type']).values.astype(np.float32)
y_test = df_test['url_type'].values.astype(np.float32)
X_query = df_query.drop(columns=['url', 'url_type']).values.astype(np.float32)
y_query = df_query['url_type'].values.astype(np.float32)

positive_samples = np.concatenate((X_train[y_train == 1], X_test[y_test == 1]), axis=0)
negative_samples = np.concatenate((X_train[y_train == 0], X_test[y_test == 0]), axis=0)

train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
test_data = lgb.Dataset(X_test, label=y_test, free_raw_data=False)

# 设置参数
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
}

n_true = df_train[df_train['url_type'] == 1].shape[0] + df_test[df_test['url_type'] == 1].shape[0]
n_false = df_train[df_train['url_type'] == 0].shape[0] + df_test[df_test['url_type'] == 0].shape[0]
n_test = len(df_test)

size = 64 * 1024
bst = None
best_bst = None
best_fpr = 1.0
epoch_each = 1
epoch_now = 0
epoch_max = 20
best_epoch = 0
best_plbf = None
best_scores = None

start_time = time.perf_counter_ns()
for i in range(int(epoch_max / epoch_each)):
    bst = lgb.train(params, train_data, epoch_each, valid_sets=[test_data], init_model=bst,
                    keep_training_booster=True)
    epoch_now += epoch_each
    bf_bytes = size - lib.lgb_url.lgb_get_model_size(bst)
    if bf_bytes <= 0:
        break
    if epoch_now < 2:
        continue

    pos_scores = bst.predict(positive_samples).tolist()
    neg_scores = bst.predict(negative_samples).tolist()
    plbf = FastPLBF_M(positive_urls_list, pos_scores, neg_scores, bf_bytes * 8.0, 50, 5)
    fpr = plbf.get_fpr()
    if best_bst is None or fpr < best_fpr:
        best_bst = bst.__copy__()
        best_fpr = fpr
        best_epoch = epoch_now
        best_plbf = plbf
        best_scores = copy.deepcopy(pos_scores)

end_time = time.perf_counter_ns()
print(f'use {(end_time - start_time) / 1000000}ms')

model_size = lib.lgb_url.lgb_get_model_size(best_bst)
print("模型在内存中所占用的大小（字节）:", model_size)
print(f"best epoch:", best_epoch)

fp_cnt = 0
query_negative = X_query
query_neg_keys = query_urls
query_neg_scores = best_bst.predict(X_query)
total = len(query_negative)

best_plbf.insert_keys(positive_urls_list, best_scores)
for key, score in zip(query_neg_keys, query_neg_scores):
    if best_plbf.contains(key, score):
        fp_cnt += 1
print(f"fpr: {float(fp_cnt) / total}")
