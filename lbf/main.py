import lightgbm as lgb
import numpy as np
import pandas as pd

import lib.bf_util
import lib.lgb_url

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


def evaluate_threshold(thresh, y_pred, y_true, bf_bytes):
    y_pred_bin = (y_pred > thresh).astype(int)
    fp_lgb = np.sum((y_pred_bin == 1) & (y_true == 0))
    bf_count = np.sum((y_pred_bin == 0) & (y_true == 1))
    fpr_bf = lib.bf_util.get_fpr(bf_count, bf_bytes)
    fpr_lgb = fp_lgb / np.sum(y_true == 0)
    fpr_lbf = fpr_lgb + (1 - fpr_lgb) * fpr_bf
    return 1 - fpr_lbf


# 初始化变量
bst = lgb.Booster(model_file='../best_bst_20480')
model_size = lib.lgb_url.lgb_get_model_size(bst)
print("模型在内存中所占用的大小（字节）:", model_size)

X_combined = np.concatenate((X_train, X_test), axis=0)
y_combined = np.concatenate((y_train, y_test), axis=0)
y_pred = bst.predict(X_combined)

for size in range(64 * 1024, 320 * 1024 + 1, 64 * 1024):
    bloom_size = size - model_size

    best_threshold = 0.01
    best_score = -np.inf
    # 遍历所有可能的阈值
    for i in range(1, 100):
        threshold = i * 0.01
        score = evaluate_threshold(threshold, y_pred, y_combined, bloom_size)
        if score > best_score:
            best_score = score
            best_threshold = threshold

    data_negative = lib.lgb_url.lgb_validate_url(bst, X_train, y_train, train_urls, X_test, y_test, test_urls,
                                                 best_threshold)
    bloom_filter = lib.lgb_url.create_bloom_filter(dataset=data_negative, bf_size=bloom_size)

    # 访问布隆过滤器的 num_bits 属性
    num_bits = bloom_filter.num_bits

    # 将比特位转换为字节（8 bits = 1 byte）
    memory_in_bytes = num_bits / 8
    print("memory of bloom filter: ", memory_in_bytes)
    print("memory of learned model: ", model_size)

    fpr = lib.lgb_url.lgb_query_url(bst, bloom_filter, X_query, y_query, query_urls, best_threshold, False)
