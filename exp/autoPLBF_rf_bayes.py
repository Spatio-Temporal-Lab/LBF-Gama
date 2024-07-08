import copy
import time

import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from openpyxl import Workbook
from sklearn.ensemble import RandomForestClassifier

import lib.lgb_url
from plbf.FastPLBF_M import FastPLBF_M

df_train = pd.read_csv('../dataset/url_train.csv')
df_test = pd.read_csv('../dataset/url_test.csv')
df_query = pd.read_csv('../dataset/url_query.csv')

train_urls = df_train['url']
test_urls = df_test['url']
query_urls = df_query['url']

positive_train_urls = df_train[df_train['url_type'] == 1]['url']
positive_test_urls = df_test[df_test['url_type'] == 1]['url']
positive_urls = pd.concat([positive_train_urls, positive_test_urls])
positive_urls_list = positive_urls.tolist()

X_train = df_train.drop(columns=['url', 'url_type']).values.astype(np.float32)
y_train = df_train['url_type'].values.astype(np.float32)
X_test = df_test.drop(columns=['url', 'url_type']).values.astype(np.float32)
y_test = df_test['url_type'].values.astype(np.float32)
X_query = df_query.drop(columns=['url', 'url_type']).values.astype(np.float32)
y_query = df_query['url_type'].values.astype(np.float32)
positive_samples = np.concatenate((X_train[y_train == 1], X_test[y_test == 1]), axis=0)
negative_samples = np.concatenate((X_train[y_train == 0], X_test[y_test == 0]), axis=0)

# 设置参数
params = {
    'n_estimators': 1,  # 每一轮训练的树的数量
    'random_state': 42,
    'max_leaf_nodes': 20,
    'warm_start': True,  # 启用 warm_start 以便在下一轮中继续训练
}

n_true = df_train[df_train['url_type'] == 1].shape[0] + df_test[df_test['url_type'] == 1].shape[0]
n_false = df_train[df_train['url_type'] == 0].shape[0] + df_test[df_test['url_type'] == 0].shape[0]
n_test = len(df_test)


best_fpr = 1.0
best_plbf = None
size = 200 * 1024


def train(n_estimators):
    global best_plbf, best_fpr
    rf = RandomForestClassifier(n_estimators=int(n_estimators), max_leaf_nodes=20, random_state=42, warm_start=True)
    rf.fit(X_train, y_train)
    model_size = lib.lgb_url.lgb_get_model_size(rf)
    bf_size = size - model_size
    print(f'size: {model_size}, {bf_size}')
    if bf_size <= 0:
        return 0

    pos_scores = rf.predict_proba(positive_samples)[:, 1].tolist()
    neg_scores = rf.predict_proba(negative_samples)[:, 1].tolist()

    plbf = FastPLBF_M(positive_urls_list, pos_scores, neg_scores, bf_size * 8.0, 50, 5)
    fpr = plbf.get_fpr()

    if best_plbf is None or fpr < best_fpr:
        best_plbf = plbf
        best_fpr = fpr

    return 1.0 - fpr


start_time = time.perf_counter_ns()
optimizer = BayesianOptimization(
    f=train,
    pbounds={'n_estimators': (1, 20)},
    random_state=42
)
optimizer.maximize(init_points=2, n_iter=8)
best_params = optimizer.max['params']
best_n_estimators = int(best_params['n_estimators'])
best_rf = RandomForestClassifier(n_estimators=best_n_estimators, max_leaf_nodes=20, random_state=42, warm_start=True)
best_rf.fit(X_train, y_train)
end_time = time.perf_counter_ns()
print(f'use {(end_time - start_time) / 1000000}ms')

fp_cnt = 0
query_negative = X_query
query_neg_keys = query_urls
query_neg_scores = best_rf.predict_proba(X_query)[:, 1]
total = len(query_negative)

for key, score in zip(query_neg_keys, query_neg_scores):
    if best_plbf.contains(key, score):
        fp_cnt += 1
print(f"fpr: {float(fp_cnt) / total}")
