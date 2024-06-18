import lightgbm as lgb
import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization

import lib.bf_util
import lib.lgb_url


df_train = pd.read_csv('dataset/url_train.csv')
df_test = pd.read_csv('dataset/url_test.csv')
df_query = pd.read_csv('dataset/url_query.csv')

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


# 设置参数
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
}
all_memory = 256 * 1024  # tweet模型大小：5 * 1024 * 1024

n_true = df_train[df_train['url_type'] == 1].shape[0] + df_test[df_test['url_type'] == 1].shape[0]
n_test = len(df_test)


def evaluate_threshold(thresh, y_pred, y_true, bf_bytes, thresh_fpr_map):
    y_pred_bin = (y_pred > thresh).astype(int)
    fp_lgb = np.sum((y_pred_bin == 1) & (y_true == 0))
    bf_count = np.sum((y_pred_bin == 0) & (y_true == 1)) * float(n_true) / n_test
    fpr_bf = lib.bf_util.get_fpr(bf_count, bf_bytes)
    fpr_lgb = fp_lgb / np.sum(y_true == 0)
    fpr_lbf = fpr_lgb + (1 - fpr_lgb) * fpr_bf
    if thresh not in thresh_fpr_map or fpr_lbf < thresh_fpr_map[thresh]:
        thresh_fpr_map[thresh] = fpr_lbf
    return 1 - fpr_lbf


bst = None
best_bst = None
best_fpr = 1.0
best_threshold = 0.5
epoch_each = 2
epoch_now = 0
epoch_max = 20
best_epoch = 0
for i in range(int(epoch_max / epoch_each)):
    bst = lgb.train(params, train_data, epoch_each, valid_sets=[test_data], init_model=bst, keep_training_booster=True)
    prediction_results = bst.predict(X_test)
    bf_bytes = all_memory - lib.lgb_url.lgb_get_model_size(bst)
    if bf_bytes <= 0:
        break

    thresh_fpr_map = {}
    # 定义搜索空间
    pbounds = {'thresh': (0.0, 1.0)}

    # 定义优化器
    optimizer = BayesianOptimization(
        f=lambda thresh: evaluate_threshold(thresh, prediction_results, y_test, bf_bytes, thresh_fpr_map),
        pbounds=pbounds,
        random_state=42,
        allow_duplicate_points=True
    )

    # 进行优化
    optimizer.maximize(n_iter=20)

    # 最优阈值
    best_thresh = optimizer.max['params']['thresh']
    fpr_lbf = thresh_fpr_map[best_thresh]
    if best_bst is None or fpr_lbf < best_fpr:
        best_bst = bst.__copy__()
        # bst.save_model('best_model.txt')
        best_threshold = best_thresh
        best_fpr = fpr_lbf
        best_epoch = epoch_now

    epoch_now += epoch_each
    thresh_fpr_map.clear()


model_size = lib.lgb_url.lgb_get_model_size(best_bst)
print("模型在内存中所占用的大小（字节）:", model_size)
print(f"best threshold:", best_threshold)
print(f"best epoch:", best_epoch)

data_negative = lib.lgb_url.lgb_validate(best_bst, X_train, y_train, train_urls, X_test, y_test, test_urls, best_threshold)
bloom_size = all_memory - model_size

bloom_filter = lib.lgb_url.create_bloom_filter(dataset=data_negative, bf_size=bloom_size)

# 访问布隆过滤器的 num_bits 属性
num_bits = bloom_filter.num_bits

# 将比特位转换为字节（8 bits = 1 byte）
memory_in_bytes = num_bits / 8
print("memory of bloom filter: ", memory_in_bytes)
print("memory of learned model: ", model_size)

fpr = lib.lgb_url.lgb_query(best_bst, bloom_filter, X_query, y_query, query_urls, best_threshold, False)
