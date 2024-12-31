import time
import mysql.connector
import pandas as pd

import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization

import lib.bf_util
import lib.lgb_url

positive_frac = 0.1
df_train = pd.read_csv('dataset/url_train.csv')
df_test = pd.read_csv('dataset/url_test.csv')
df_query = pd.read_csv('dataset/url_query.csv')
df_sample = pd.read_csv('dataset/url_sample' + str(positive_frac) + '.csv')

train_urls = df_train['url']
test_urls = df_test['url']
query_urls = df_query['url']

X_train = df_train.drop(columns=['url', 'url_type']).values.astype(np.float32)
y_train = df_train['url_type'].values.astype(np.float32)
X_test = df_test.drop(columns=['url', 'url_type']).values.astype(np.float32)
y_test = df_test['url_type'].values.astype(np.float32)
X_query = df_query.drop(columns=['url', 'url_type']).values.astype(np.float32)
y_query = df_query['url_type'].values.astype(np.float32)
X_sample = df_sample.drop(columns=['url', 'url_type']).values.astype(np.float32)
y_sample = df_sample['url_type'].values.astype(np.float32)

train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
test_data = lgb.Dataset(X_test, label=y_test, free_raw_data=False)

# 设置参数
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'verbose': -1
}

n_true = df_train[df_train['url_type'] == 1].shape[0] + df_test[df_test['url_type'] == 1].shape[0]
n_false = df_train[df_train['url_type'] == 0].shape[0] + df_test[df_test['url_type'] == 0].shape[0]
n_test = len(df_test)

size = 128 * 1024
# for size in range(64 * 1024, 320 * 1024 + 1, 64 * 1024):
bst = None
model_sizes = []
best_fpr = 1.0
best_threshold = 0.5
best_epoch = 0

start_time = time.perf_counter_ns()
evaluate_epoch_count = 5
cur_epoch = evaluate_epoch_count


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
        fpr_bf = lib.bf_util.get_fpr(bf_count / positive_frac, bf_bytes)
        fpr_lgb = fp / n_false
        fpr_lbf = fpr_lgb + (1 - fpr_lgb) * fpr_bf

        if fpr_lbf < best_fpr_lbf:
            best_thresh = thresh
            best_fpr_lbf = fpr_lbf

    # print(f'best thresh = {best_thresh} and best fpr = {best_fpr_lbf}')
    return best_thresh, best_fpr_lbf


bst = lgb.Booster(params=params, train_set=train_data)
for i in range(evaluate_epoch_count):
    bst.update(train_data)
    size_temp = lib.lgb_url.lgb_get_model_size(bst)
    if size_temp > size:
        break
    model_sizes.append(size_temp)
    bf_bytes = size - size_temp
    sample_pred = bst.predict(X_sample, num_iteration=i + 1)
    best_thresh, best_fpr_lbf = evaluate_thresholds(sample_pred, y_sample, bf_bytes)
    if best_fpr_lbf < best_fpr:
        best_threshold = best_thresh
        best_fpr = best_fpr_lbf
        best_epoch = i + 1


def fit_linear_function(model_sizes, evaluate_epoch_count):
    x = np.arange(1, evaluate_epoch_count + 1)
    y = np.array(model_sizes[:evaluate_epoch_count])
    coefficients = np.polyfit(x, y, 1)
    a = coefficients[0]  # 斜率
    b = coefficients[1]  # 截距
    return a, b


# model_size = a * epoch + b
a, b = fit_linear_function(model_sizes, evaluate_epoch_count)
epoch_max = int((size - b) / a)
print(epoch_max)

fpr_map = dict()


def objective(query_epoch):
    global cur_epoch, epoch_max
    query_epoch = int(query_epoch)
    if query_epoch > epoch_max:
        return -float('inf')

    if fpr_map.__contains__(query_epoch):
        return fpr_map[query_epoch]

    while cur_epoch < query_epoch:
        cur_epoch += 1
        bst.update(train_data)
        size_temp = lib.lgb_url.lgb_get_model_size(bst)
        if size_temp > size:
            epoch_max = cur_epoch - 1
            break
        model_sizes.append(size_temp)

    model_size = model_sizes[query_epoch - 1]  # Model size at `num_iteration`
    bf_bytes = size - model_size

    sample_pred = bst.predict(X_sample, num_iteration=query_epoch)
    best_thresh, best_fpr_lbf = evaluate_thresholds(sample_pred, y_sample, bf_bytes)

    global best_fpr, best_threshold, best_epoch
    if best_fpr_lbf < best_fpr:
        best_threshold = best_thresh
        best_fpr = best_fpr_lbf
        best_epoch = query_epoch

    fpr_map[query_epoch] = best_fpr_lbf

    return -best_fpr_lbf


# 使用贝叶斯优化来寻找最佳 epoch
def optimize_epochs():
    l_bound = evaluate_epoch_count + 1
    r_bound = epoch_max
    pbounds = {'query_epoch': (l_bound, r_bound)}  # 设置 epoch 范围
    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        verbose=0,
        random_state=42,
    )

    if r_bound - l_bound > 5:
        optimizer.maximize(init_points=5, n_iter=min(20, (r_bound - l_bound) // 2))
    else:
        optimizer.maximize(init_points=r_bound - l_bound, n_iter=1)


optimize_epochs()
best_bst = lgb.Booster(model_str=bst.model_to_string(num_iteration=best_epoch))

model_size = lib.lgb_url.lgb_get_model_size(best_bst)

data_negative = lib.lgb_url.lgb_validate_url(best_bst, X_train, y_train, train_urls, X_test, y_test, test_urls,
                                             best_threshold)
bloom_size = size - model_size

bloom_filter = lib.lgb_url.create_bloom_filter(dataset=data_negative, bf_size=bloom_size)
end_time = time.perf_counter_ns()
print(f'use {(end_time - start_time) / 1000000}ms')

for query_positive_ratio in [0.2, 0.4, 0.6, 0.8]:
    # 读取CSV文件
    df = pd.read_csv("dataset/url.csv", delimiter=',')  # 替换为你的CSV文件路径
    positive_data = df[df['url_type'] == 1]
    negative_data = df[df['url_type'] == 0]
    query_total = 10000
    num_positive = int(query_total * query_positive_ratio)
    num_negative = query_total - num_positive
    positive_sample = positive_data.sample(n=num_positive, random_state=42)
    negative_sample = negative_data.sample(n=num_negative, random_state=42)
    sampled_data = pd.concat([positive_sample, negative_sample])
    sampled_urls = sampled_data['url'].tolist()
    y = sampled_data['url_type'].values.astype(np.float32)
    X = sampled_data.drop(columns=['url', 'url_type']).values.astype(np.float32)

    # 连接到MySQL数据库
    db_connection = mysql.connector.connect(
        host="localhost",  # MySQL服务器地址
        user="xxx",  # 替换为你的MySQL用户名
        password="xxx",  # 替换为你的MySQL密码
        database="xxx"  # 替换为你的数据库名
    )

    cursor = db_connection.cursor()

    query_start_time = time.time()
    binary_predictions = (best_bst.predict(X) > best_threshold).astype(int)
    # 筛选出符合条件的 URL
    urls_filtered = [
        url for i, (url, prediction) in enumerate(zip(sampled_urls, binary_predictions))
        if prediction == 1 or url in bloom_filter
    ]
    # 批量执行查询
    single_query = "SELECT url FROM urls WHERE url = %s"
    cursor.executemany(single_query, [(url,) for url in urls_filtered])
    # 获取所有查询的结果
    result = cursor.fetchall()
    query_end_time = time.time()
    total_query_time = query_end_time - query_start_time
    # 输出总查询时间
    print(f"所有URL查询的总时间: {total_query_time:.4f} 秒")

    # 关闭连接
    cursor.close()
    db_connection.close()

    # 连接到MySQL数据库
    db_connection = mysql.connector.connect(
        host="localhost",  # MySQL服务器地址
        user="xxx",  # 替换为你的MySQL用户名
        password="xxx",  # 替换为你的MySQL密码
        database="xxx"  # 替换为你的数据库名
    )
    cursor = db_connection.cursor()
    query_start_time = time.time()
    single_query = "SELECT url FROM urls WHERE url = %s"
    cursor.executemany(single_query, [(url,) for url in sampled_urls])
    # 获取所有查询的结果
    result = cursor.fetchall()
    query_end_time = time.time()
    total_query_time = query_end_time - query_start_time
    # 输出总查询时间
    print(f"所有URL查询的总时间: {total_query_time:.4f} 秒")
    # 关闭连接
    cursor.close()
    db_connection.close()
