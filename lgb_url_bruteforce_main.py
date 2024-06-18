import lightgbm as lgb
import numpy as np
import pandas as pd

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
all_memory = 32 * 1024  # tweet模型大小：5 * 1024 * 1024

n_true = df_train[df_train['url_type'] == 1].shape[0] + df_test[df_test['url_type'] == 1].shape[0]
n_test = len(df_test)


def evaluate_thresholds(prediction_results, y_true, bf_bytes):
    sorted_indices = np.argsort(prediction_results)
    sorted_predictions = prediction_results[sorted_indices]
    sorted_true = y_true[sorted_indices]

    print(sorted_predictions)

    total_positives = np.sum(sorted_true)
    print(f'total positives = {total_positives}')
    total_negatives = len(sorted_true) - total_positives
    print(f'total negatives = {total_negatives}')

    # fp = 0
    # tp = total_positives
    fp = total_negatives
    tp = 0
    best_thresh = 0
    best_fpr_lbf = 1.0

    unique_sorted_predictions = np.unique(sorted_predictions)

    j = 0
    for i in range(len(unique_sorted_predictions)):
        thresh = unique_sorted_predictions[i]

        while j < len(sorted_predictions) and sorted_predictions[j] == thresh:
            if sorted_true[j] == 1:
                tp += 1
            else:
                fp -= 1
            j += 1

        # bf_count = tp * float(n_true) / n_test
        bf_count = tp
        fpr_bf = lib.bf_util.get_fpr(bf_count, bf_bytes)
        fpr_lgb = fp / total_negatives
        fpr_lbf = fpr_lgb + (1 - fpr_lgb) * fpr_bf

        if fpr_lbf < best_fpr_lbf:
            best_thresh = thresh
            best_fpr_lbf = fpr_lbf

    print(f'best thresh = {best_thresh} and best fpr = {best_fpr_lbf}')
    return best_thresh, best_fpr_lbf


bst = None
best_bst = None
best_fpr = 1.0
best_threshold = 0.5
epoch_each = 1
epoch_now = 0
epoch_max = 20
best_epoch = 0
for i in range(int(epoch_max / epoch_each)):
    bst = lgb.train(params, train_data, epoch_each, valid_sets=[test_data], init_model=bst, keep_training_booster=True)
    bf_bytes = all_memory - lib.lgb_url.lgb_get_model_size(bst)
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


model_size = lib.lgb_url.lgb_get_model_size(best_bst)
print("模型在内存中所占用的大小（字节）:", model_size)
print(f"best threshold:", best_threshold)
print(f"best epoch:", best_epoch)

data_negative = lib.lgb_url.lgb_validate(best_bst, X_train, y_train, train_urls, X_test, y_test, test_urls, best_threshold)
print(f"{len(data_negative)} insert into bloom filter")
bloom_size = all_memory - model_size

bloom_filter = lib.lgb_url.create_bloom_filter(dataset=data_negative, bf_size=bloom_size)

# 访问布隆过滤器的 num_bits 属性
num_bits = bloom_filter.num_bits

# 将比特位转换为字节（8 bits = 1 byte）
memory_in_bytes = num_bits / 8
print("memory of bloom filter: ", memory_in_bytes)
print("memory of learned model: ", model_size)

fpr = lib.lgb_url.lgb_query(best_bst, bloom_filter, X_query, y_query, query_urls, best_threshold, False)
