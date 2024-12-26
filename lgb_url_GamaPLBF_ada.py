import copy
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization

import lib.bf_util
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

for size in range(64 * 1024, 320 * 1024 + 1, 64 * 1024):
    bst = None
    model_sizes = []
    best_fpr = 1.0
    best_plbf = None
    best_epoch = 0

    start_time = time.perf_counter_ns()
    evaluate_epoch_count = 5
    cur_epoch = evaluate_epoch_count

    bst = lgb.Booster(params=params, train_set=train_data)
    for i in range(evaluate_epoch_count):
        bst.update(train_data)
        size_temp = lib.lgb_url.lgb_get_model_size(bst)
        if size_temp > size:
            break
        model_sizes.append(size_temp)
        if i == 0:
            continue
        bf_bytes = size - size_temp

        pos_scores = bst.predict(positive_samples).tolist()
        neg_scores = bst.predict(negative_samples).tolist()
        plbf = FastPLBF_M(positive_urls_list, pos_scores, neg_scores, bf_bytes * 8.0, 50, 5)
        cur_fpr = plbf.get_fpr()

        if cur_fpr < best_fpr:
            best_plbf = copy.deepcopy(plbf)
            best_fpr = cur_fpr
            best_epoch = i + 1


    def fit_linear_function(model_sizes, evaluate_epoch_count):
        x = np.arange(1, evaluate_epoch_count + 1)
        y = np.array(model_sizes[:evaluate_epoch_count])
        coefficients = np.polyfit(x, y, 1)
        a = coefficients[0]  # 斜率
        b = coefficients[1]  # 截距
        return a, b


    # model_size = a * epoch + b
    a, b = fit_linear_function(model_sizes, evaluate_epoch_count)
    epoch_max = int((size - b) / a)
    print(epoch_max)

    fpr_map = dict()


    def objective(query_epoch):
        global cur_epoch, epoch_max
        query_epoch = int(query_epoch)
        if query_epoch > epoch_max:
            return -float('inf')

        if fpr_map.__contains__(query_epoch):
            return fpr_map[query_epoch]

        while cur_epoch < query_epoch:
            cur_epoch += 1
            bst.update(train_data)
            size_temp = lib.lgb_url.lgb_get_model_size(bst)
            if size_temp > size:
                epoch_max = cur_epoch - 1
                break
            model_sizes.append(size_temp)

        model_size = model_sizes[query_epoch - 1]  # Model size at `num_iteration`
        bf_bytes = size - model_size

        # sample_pred = bst.predict(X_sample, num_iteration=query_epoch)
        # best_thresh, best_fpr_lbf = evaluate_thresholds(sample_pred, y_sample, bf_bytes)
        pos_scores = bst.predict(positive_samples, num_iteration=query_epoch).tolist()
        neg_scores = bst.predict(negative_samples, num_iteration=query_epoch).tolist()
        plbf = FastPLBF_M(positive_urls_list, pos_scores, neg_scores, bf_bytes * 8.0, 50, 5)
        cur_fpr = plbf.get_fpr()

        global best_fpr, best_plbf, best_epoch
        if cur_fpr < best_fpr:
            best_plbf = copy.deepcopy(plbf)
            best_fpr = cur_fpr
            best_epoch = i + 1
        return -cur_fpr

        # global best_fpr, best_threshold, best_epoch
        # if best_fpr_lbf < best_fpr:
        #     best_threshold = best_thresh
        #     best_fpr = best_fpr_lbf
        #     best_epoch = query_epoch
        #
        # fpr_map[query_epoch] = best_fpr_lbf
        #
        # return -best_fpr_lbf


    # 使用贝叶斯优化来寻找最佳 epoch
    def optimize_epochs():
        l_bound = evaluate_epoch_count + 1
        r_bound = epoch_max
        pbounds = {'query_epoch': (l_bound, r_bound)}  # 设置 epoch 范围
        optimizer = BayesianOptimization(
            f=objective,
            pbounds=pbounds,
            verbose=0,
            random_state=42,
            allow_duplicate_points=True
        )

        if r_bound - l_bound > 5:
            optimizer.maximize(init_points=5, n_iter=min(20, (r_bound - l_bound) // 2))
        else:
            optimizer.maximize(init_points=r_bound - l_bound, n_iter=1)


    optimize_epochs()
    best_bst = lgb.Booster(model_str=bst.model_to_string(num_iteration=best_epoch))
    end_time = time.perf_counter_ns()
    print(f'use {(end_time - start_time) / 1000000}ms')

    model_size = lib.lgb_url.lgb_get_model_size(best_bst)
    print("模型在内存中所占用的大小（字节）:", model_size)
    print(f"best epoch:", best_epoch)

    pos_scores = best_bst.predict(positive_items).tolist()
    # neg_scores = best_bst.predict(negative_items).tolist()
    # bf_bytes = size - lib.lgb_url.lgb_get_model_size(best_bst)
    # best_plbf = FastPLBF_M(positive_urls_list, pos_scores, neg_scores, bf_bytes * 8.0, 50, 5)

    fp_cnt = 0
    query_negative = X_query
    query_neg_keys = query_urls
    query_neg_scores = best_bst.predict(X_query)
    total = len(query_negative)

    best_plbf.insert_keys(positive_urls_list, pos_scores)
    for key, score in zip(query_neg_keys, query_neg_scores):
        if best_plbf.contains(key, score):
            fp_cnt += 1
    print(f"fpr: {float(fp_cnt) / total}")
