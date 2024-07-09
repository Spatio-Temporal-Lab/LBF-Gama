import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import lib.network
import lib.data_processing
import lib.bf_util
import lib.lgb_url

data_train = pd.read_csv('dataset/tweet/tweet_train.csv')
data_test = pd.read_csv('dataset/tweet/tweet_test.csv')
data_query = pd.read_csv('dataset/tweet/tweet_query.csv')

word_dict, region_dict = lib.data_processing.loading_embedding("tweet")


def yelp_embedding(data_train, word_dict=word_dict, region_dict=region_dict):
    data_train['keywords'] = data_train['keywords'].str.split(' ')
    data_train = data_train.explode('keywords')
    data_train = data_train.reset_index(drop=True)
    data_train['keywords'] = data_train['keywords'].astype(str)
    data_train['keywords'] = data_train['keywords'].apply(str.lower)

    true_num = data_train[data_train['is_in'] == 1].shape[0]
    false_num = data_train[data_train['is_in'] == 0].shape[0]
    insert = pd.DataFrame()
    insert = data_train.apply(lib.network.insert, axis=1)

    # region embedding
    data_train['region'] = data_train.apply(lib.network.region_mapping, axis=1, args=(region_dict,))
    data_train.drop(columns=['lat', 'lon'], inplace=True)

    # time embedding
    data_train['timestamp'] = data_train['timestamp'].apply(lib.network.time_embedding)

    # keywords embedding
    data_train['keywords'] = data_train['keywords'].apply(lib.network.keywords_embedding, args=(word_dict,))

    # 生成一个用于神经网络输入的dataframe:embedding
    embedding = pd.DataFrame()
    embedding['embedding'] = data_train.apply(lib.network.to_embedding, axis=1)
    # print(embedding)
    y = data_train['is_in']
    del data_train
    X = pd.DataFrame(embedding['embedding'].apply(pd.Series))
    # print(X)
    return X, y, insert, true_num, false_num


X_train, y_train, train_insert, train_true, train_false = yelp_embedding(data_train, word_dict=word_dict,
                                                                         region_dict=region_dict)
X_test, y_test, test_insert, test_true, test_false = yelp_embedding(data_test, word_dict=word_dict,
                                                                    region_dict=region_dict)
X_query, y_query, query_insert, query_true, query_false = yelp_embedding(data_query, word_dict=word_dict,
                                                                         region_dict=region_dict)
print(query_insert)
train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
test_data = lgb.Dataset(X_test, label=y_test, free_raw_data=False)

n_true = train_true + test_true
n_false = test_false + train_false

n_test = test_true + test_false

# 设置参数
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'verbose': -1
}


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

    print(f'best thresh = {best_thresh} and best fpr = {best_fpr_lbf}')
    return best_thresh, best_fpr_lbf


# def evaluate_thresholds(prediction_results, y_true, bf_bytes):
#     sorted_indices = np.argsort(prediction_results)
#     sorted_predictions = prediction_results[sorted_indices]
#     sorted_true = y_true[sorted_indices]
#
#     fp = n_false
#     tp = 0
#     best_thresh = 0
#     best_fpr_lbf = 1.0
#
#     unique_sorted_predictions = np.unique(sorted_predictions)
#
#     j = 0
#     for i in range(len(unique_sorted_predictions)):
#         thresh = unique_sorted_predictions[i]
#
#         while j < len(sorted_predictions) and sorted_predictions[j] == thresh:
#             if sorted_true[j] == 1:
#                 tp += 1
#             else:
#                 fp -= 1
#             j += 1
#
#         bf_count = tp
#         fpr_bf = lib.bf_util.get_fpr(bf_count, bf_bytes)
#         fpr_lgb = fp / n_false
#         fpr_lbf = fpr_lgb + (1 - fpr_lgb) * fpr_bf
#
#         if fpr_lbf < best_fpr_lbf:
#             best_thresh = thresh
#             best_fpr_lbf = fpr_lbf
#
#     print(f'best thresh = {best_thresh} and best fpr = {best_fpr_lbf}')
#     return best_thresh, best_fpr_lbf


# def evaluate_thresholds(prediction_results, y_true, bf_bytes):
#     # 量化预测值并转换为整数
#     quantized_predictions = np.round(prediction_results * 1000).astype(int)
#
#     # # 获取最大值和最小值以确定桶的范围
#     max_pred = np.max(quantized_predictions)
#     min_pred = np.min(quantized_predictions)
#     #
#     # # 初始化桶的数量
#     num_buckets = max_pred - min_pred + 1
#     tp_count = np.zeros(num_buckets, dtype=int)
#     fp_count = np.zeros(num_buckets, dtype=int)
#
#     indices = quantized_predictions - min_pred
#     np.add.at(tp_count, indices[y_true == 1], 1)
#     np.add.at(fp_count, indices[y_true == 0], 1)
#     best_thresh = 0
#     best_fpr_lbf = 1.0
#
#     tp = 0
#     fp = n_false
#     for i in range(num_buckets):
#         tp += tp_count[i]
#         fp -= fp_count[i]
#
#         bf_count = tp
#         fpr_bf = lib.bf_util.get_fpr(bf_count, bf_bytes)
#         fpr_lgb = fp / n_false
#         fpr_lbf = fpr_lgb + (1 - fpr_lgb) * fpr_bf
#
#         if fpr_lbf < best_fpr_lbf:
#             best_thresh = (i + min_pred) / 1000.0
#             best_fpr_lbf = fpr_lbf
#
#     print(f'best thresh = {best_thresh} and best fpr = {best_fpr_lbf}')
#     return best_thresh, best_fpr_lbf


start_time = time.perf_counter_ns()
initial_size = 64 * 1024
max_size = 320 * 1024

# 循环，从32开始，每次乘以2，直到512
size = initial_size
while size <= max_size:
    print(f'size {size}')
    bst = None
    best_bst = None
    best_fpr = 1.0
    best_threshold = 0.5
    epoch_each = 1
    epoch_now = 0
    epoch_max = 20
    best_epoch = 0
    for i in range(int(epoch_max / epoch_each)):
        bst = lgb.train(params, train_data, epoch_each, valid_sets=[test_data], init_model=bst,
                        keep_training_booster=True)
        bf_bytes = size - lib.lgb_url.lgb_get_model_size(bst)
        if bf_bytes <= 0:
            break
        # prediction_results = bst.predict(X_test)

        # 对训练集进行预测
        train_pred = bst.predict(X_train)
        test_pred = bst.predict(X_test)

        # 拼接预测结果
        all_predictions = np.concatenate([train_pred, test_pred])
        all_true_labels = np.concatenate([y_train, y_test])
        best_thresh, best_fpr_lbf = evaluate_thresholds(all_predictions, all_true_labels, bf_bytes)

        # best_thresh, best_fpr_lbf = evaluate_thresholds(prediction_results, y_test, bf_bytes)

        # 保存最佳模型
        if best_bst is None or best_fpr_lbf < best_fpr:
            best_bst = bst.__copy__()
            best_threshold = best_thresh
            best_fpr = best_fpr_lbf
            best_epoch = epoch_now

        epoch_now += epoch_each

    end_time = time.perf_counter_ns()
    print(f'use {(end_time - start_time) / 1000000}ms')

    model_size = lib.lgb_url.lgb_get_model_size(best_bst)
    print("模型在内存中所占用的大小（字节）:", model_size)
    print(f"best threshold:", best_threshold)
    print(f"best epoch:", best_epoch)

    data_negative = lib.lgb_url.lgb_validate_url(best_bst, X_train, y_train, train_insert, X_test, y_test, test_insert,
                                                 best_threshold)
    print(f"{len(data_negative)} insert into bloom filter")
    bloom_size = size - model_size

    bloom_filter = lib.lgb_url.create_bloom_filter(dataset=data_negative, bf_size=bloom_size)

    # 访问布隆过滤器的 num_bits 属性
    num_bits = bloom_filter.num_bits

    # 将比特位转换为字节（8 bits = 1 byte）
    memory_in_bytes = num_bits / 8
    print("memory of bloom filter: ", memory_in_bytes)
    print("memory of learned model: ", model_size)

    fpr = lib.lgb_url.lgb_query_url(best_bst, bloom_filter, X_query, y_query, query_insert, best_threshold, False)
    size = size + 64 * 1024
