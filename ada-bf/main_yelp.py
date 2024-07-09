import lightgbm as lgb
import numpy as np
import pandas as pd
import lib.lgb_url
import ada_bf
import disjoint_ada_bf

import lib.network
import lib.data_processing
import lib.lgb_url
import lib.bf_util


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
    #print(embedding)
    y = data_train['is_in']
    del data_train
    X = pd.DataFrame(embedding['embedding'].apply(pd.Series))
    #print(X)
    return X, y, insert


X_train, y_train, train_insert = yelp_embedding(data_train, word_dict=word_dict, region_dict=region_dict)
X_test, y_test, test_insert = yelp_embedding(data_test, word_dict=word_dict, region_dict=region_dict)
X_query, y_query, query_insert = yelp_embedding(data_query, word_dict=word_dict, region_dict=region_dict)

n_true = data_train[data_train['is_in'] == 1].shape[0] + data_test[data_test['is_in'] == 1].shape[0]
n_test = len(data_test)
# 清理内存



bst = lgb.Booster(model_file='../best_bst_20480')

y_pred_train = bst.predict(X_train)
y_pred_test = bst.predict(X_test)

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

# 初始化变量
model_size = lib.lgb_url.lgb_get_model_size(bst)
print("模型在内存中所占用的大小（字节）:", model_size)

initial_size = 32 * 1024
max_size = 512 * 1024

# 循环，从32开始，每次乘以2，直到512
size = initial_size
while size <= max_size:
    bloom_size = size - model_size
    # ada_bf.run(
    #     num_group_min=8,
    #     num_group_max=12,
    #     R_sum=bloom_size*8,
    #     c_min=1.6,
    #     c_max=2.5,
    #     path='url_results.csv',
    #     model=bst,
    #     X_query=X_query,
    #     y_query=y_query,
    #     query_urls=query_insert
    # )
    disjoint_ada_bf.run(
        num_group_min=8,
        num_group_max=12,
        R_sum=bloom_size*8,
        c_min=1.6,
        c_max=2.5,
        path='url_results.csv',
        model=bst,
        X_query=X_query,
        y_query=y_query,
        query_urls=query_insert
    )
    size *= 2
