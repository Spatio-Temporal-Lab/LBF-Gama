import lightgbm as lgb
import numpy as np
import pandas as pd

import lib.lgb_url
import learn_bf
import lib.network
import lib.data_processing

data_train = pd.read_csv('../dataset/tweet/tweet_train.csv')
data_test = pd.read_csv('../dataset/tweet/tweet_test.csv')
data_query = pd.read_csv('../dataset/tweet/tweet_query.csv')

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

n_true = train_true + test_true
n_false = test_false + train_false

n_test = test_true + test_false

# 清理内存


# 3. 划分训练集和测试集
X_train = X_train.values.astype(np.float32)
X_test = X_test.values.astype(np.float32)
y_train = y_train.values.astype(np.float32)
y_test = y_test.values.astype(np.float32)
# 4. 创建 LightGBM 数据集
train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data, free_raw_data=False)
query_data = lgb.Dataset(X_query, label=y_query, free_raw_data=False)

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
all_results = pd.concat([train_results, test_results])
all_results.to_csv('url_results.csv', index=False)

# 初始化变量
model_size = lib.lgb_url.lgb_get_model_size(bst)
print("模型在内存中所占用的大小（字节）:", model_size)

for size in range(64 * 1024, 320 * 1024 + 1, 64 * 1024):
    bloom_size = size - model_size
    learn_bf.run(
        R_sum=bloom_size * 8,
        path='url_results.csv',
        model=bst,
        X_query=X_query,
        y_query=y_query,
        query_urls=query_insert
    )
