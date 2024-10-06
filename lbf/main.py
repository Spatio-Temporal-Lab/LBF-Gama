import lightgbm as lgb
import numpy as np
import pandas as pd
import sys
import os

lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../lib'))
sys.path.insert(0, lib_path)

# 从 lib 包中导入 lgb_url 模块

from lib import lgb_url
import learn_bf

df_train = pd.read_csv('../Train_COD.csv')
df_test = pd.read_csv('../Test_COD.csv')
df_query = pd.read_csv('../Query_COD.csv')

train_urls = df_train['objID']
test_urls = df_test['objID']
query_urls = df_query['objID']

X_train = df_train.drop(columns=['type']).values.astype(np.float32)
y_train = df_train['type'].values.astype(np.float32)
X_test = df_test.drop(columns=['type']).values.astype(np.float32)
y_test = df_test['type'].values.astype(np.float32)
X_query = df_query.drop(columns=['type']).values.astype(np.float32)
y_query = df_query['type'].values.astype(np.float32)

train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
test_data = lgb.Dataset(X_test, label=y_test, free_raw_data=False)
n_true = df_train[df_train['type'] == 1].shape[0] + df_test[df_test['type'] == 1].shape[0]
n_test = len(df_test)

bst = lgb.Booster(model_file='../best_bst_204800')

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
model_size = lgb_url.lgb_get_model_size(bst)
print("模型在内存中所占用的大小（字节）:", model_size)

for size in range(int(1* 1024 * 1024), int(2.5*1024 * 1024 + 1), int(0.5*1024 * 1024)):
    bloom_size = size - model_size
    learn_bf.run(
        R_sum=bloom_size*8,
        path='url_results.csv',
        model=bst,
        X_query=X_query,
        y_query=y_query,
        query_urls=query_urls
    )

