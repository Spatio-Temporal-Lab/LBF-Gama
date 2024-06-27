import lightgbm as lgb
import numpy as np
import pandas as pd

import lib.lgb_url
import ada_bf
import disjoint_ada_bf

df_train = pd.read_csv('../dataset/url_train.csv')
df_test = pd.read_csv('../dataset/url_test.csv')
df_query = pd.read_csv('../dataset/url_query.csv')

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
n_true = df_train[df_train['url_type'] == 1].shape[0] + df_test[df_test['url_type'] == 1].shape[0]
n_test = len(df_test)

bst = lgb.Booster(model_file='../best_bst_20480')

y_pred_train = bst.predict(X_train)
y_pred_test = bst.predict(X_test)

train_results = pd.DataFrame({
    'url': train_urls,
    'label': y_train,
    'score': y_pred_train
})

test_results = pd.DataFrame({
    'url': test_urls,
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
    ada_bf.run(
        num_group_min=8,
        num_group_max=12,
        R_sum=bloom_size*8,
        c_min=1.6,
        c_max=2.5,
        path='url_results.csv',
        model=bst,
        X_query=X_query,
        y_query=y_query,
        query_urls=query_urls
    )
    # disjoint_ada_bf.run(
    #     num_group_min=8,
    #     num_group_max=12,
    #     R_sum=bloom_size*8,
    #     c_min=1.6,
    #     c_max=2.5,
    #     path='url_results.csv',
    #     model=bst,
    #     X_query=X_query,
    #     y_query=y_query,
    #     query_urls=query_urls
    # )
    size *= 2
