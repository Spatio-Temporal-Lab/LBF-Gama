import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import lib.network
import lib.data_processing
import lib.lgb_url
from plbf.FastPLBF_M import FastPLBF_M

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
    positive_insert = insert[data_train['is_in'] == 1]

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
    return X, y, insert, true_num, false_num, positive_insert


X_train, y_train, train_insert, train_true, train_false, train_positive_insert = yelp_embedding(data_train,
                                                                                                word_dict=word_dict,
                                                                                                region_dict=region_dict)
X_test, y_test, test_insert, test_true, test_false, test_positive_insert = yelp_embedding(data_test,
                                                                                          word_dict=word_dict,
                                                                                          region_dict=region_dict)
X_query, y_query, query_insert, query_true, query_false, query_positive_insert = yelp_embedding(data_query,
                                                                                                word_dict=word_dict,
                                                                                                region_dict=region_dict)
print(train_positive_insert)
train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
test_data = lgb.Dataset(X_test, label=y_test, free_raw_data=False)

n_true = train_true + test_true
n_false = test_false + train_false

n_test = test_true + test_false

positive_samples = np.concatenate((X_train[y_train == 1], X_test[y_test == 1]), axis=0)
negative_samples = np.concatenate((X_train[y_train == 0], X_test[y_test == 0]), axis=0)


# 合并正类样本的URLs
positive_urls = pd.concat([train_positive_insert, test_positive_insert])

# 转换成list
positive_urls_list = positive_urls.tolist()

# 设置参数
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
}

size = 64 * 1024
bst = None
best_bst = None
best_fpr = 1.0
epoch_each = 1
epoch_now = 0
epoch_max = 20
best_epoch = 0
best_plbf = None

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

end_time = time.perf_counter_ns()
print(f'use {(end_time - start_time) / 1000000}ms')

model_size = lib.lgb_url.lgb_get_model_size(best_bst)
print("模型在内存中所占用的大小（字节）:", model_size)
print(f"best epoch:", best_epoch)

fp_cnt = 0
query_negative = X_query
query_neg_keys = query_insert
query_neg_scores = best_bst.predict(X_query)
total = len(query_negative)

for key, score in zip(query_neg_keys, query_neg_scores):
    if best_plbf.contains(key, score):
        fp_cnt += 1
print(f"fpr: {float(fp_cnt) / total}")
