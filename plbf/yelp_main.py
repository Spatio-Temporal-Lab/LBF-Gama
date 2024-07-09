import lightgbm as lgb
import numpy as np
import pandas as pd
import lib.network
import lib.data_processing
import lib.lgb_url
from plbf import FastPLBF_M


data_train = pd.read_csv('../dataset/yelp/yelp_train.csv')
data_test = pd.read_csv('../dataset/yelp/yelp_test.csv')
data_query = pd.read_csv('../dataset/yelp/yelp_query.csv')

word_dict, region_dict = lib.data_processing.loading_embedding("yelp")


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


bst = lgb.Booster(model_file='../best_bst_20480')

y_pred_train = bst.predict(X_train)
y_pred_test = bst.predict(X_test)
y_pred_query = bst.predict(X_query)

train_results = pd.DataFrame({
    'url': train_insert,
    'label': y_train,
    'score': y_pred_train
})

test_results = pd.DataFrame({
    'url': test_insert,
    'label': y_test,
    'score': y_pred_test
})
all_results = pd.concat([train_results, test_results])
all_results.to_csv('url_results.csv', index=False)
query_results = pd.DataFrame({
    'url': query_insert,
    'label': y_query,
    'score': y_pred_query
})
query_results.to_csv('query_results.csv', index=False)
# 初始化变量
model_size = lib.lgb_url.lgb_get_model_size(bst)
print("模型在内存中所占用的大小（字节）:", model_size)

for size in range(64 * 1024, 320 * 1024 + 1, 64 * 1024):
    bloom_size = size - model_size
    print(bloom_size)
    bloom_size = bloom_size * 8.0
    FastPLBF_M.run(
        path='url_results.csv',
        query_path='query_results.csv',
        M=bloom_size,
        N=50,
        k=5,
    )
