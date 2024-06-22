import lightgbm as lgb
import pandas as pd
import lib.network
import lib.data_processing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np
import pickle
import lib.lgb_url
import lib.bf_util
from bayes_opt import BayesianOptimization

'''
df = pd.read_csv('dataset/yelp/query_data.csv')

# 将标签（1）和标签（0）分别过滤出来
df_1 = df[df['is_in'] == 1]
df_0 = df[df['is_in'] == 0]

# 从标签为'b'的样本中随机抽取和's'标签相同数量的样本
df_1_sample = df_1.sample(frac=0.8, random_state=42)
df_0_sample = df_0.sample(frac=0.8, random_state=42)

# 合并得到训练集+测试集
df_train_test = pd.concat([df_1_sample, df_0_sample])

# 剩下的标签为'b'的样本作为查询集
df_query = df_0.drop(df_0_sample.index)

df_train, df_test = train_test_split(df_train_test, test_size=0.2, random_state=42)



df_train.to_csv('dataset/yelp/yelp_train.csv', index=False)
df_test.to_csv('dataset/yelp/yelp_test.csv', index=False)
df_query.to_csv('dataset/yelp/yelp_query.csv', index=False)
'''

data_train = pd.read_csv('dataset/yelp/yelp_train.csv')
data_test = pd.read_csv('dataset/yelp/yelp_test.csv')
data_query = pd.read_csv('dataset/yelp/yelp_query.csv')

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


# 3. 划分训练集和测试集
X_train = X_train.values.astype(np.float32)
X_test = X_test.values.astype(np.float32)
y_train = y_train.values.astype(np.float32)
y_test = y_test.values.astype(np.float32)
# 4. 创建 LightGBM 数据集
train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data, free_raw_data=False)

# 设置参数
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
}
all_memory = 512 * 1024  # tweet模型大小：5 * 1024 * 1024


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

data_negative = lib.lgb_url.lgb_validate_url(best_bst, X_train, y_train, train_insert, X_test, y_test, test_insert,
                                             best_threshold)
print(f"{len(data_negative)} insert into bloom filter")
bloom_size = all_memory - model_size

bloom_filter = lib.lgb_url.create_bloom_filter(dataset=data_negative, bf_size=bloom_size)

# 访问布隆过滤器的 num_bits 属性
num_bits = bloom_filter.num_bits

# 将比特位转换为字节（8 bits = 1 byte）
memory_in_bytes = num_bits / 8
print("memory of bloom filter: ", memory_in_bytes)
print("memory of learned model: ", model_size)

fpr = lib.lgb_url.lgb_query_url(best_bst, bloom_filter, X_query, y_query, query_insert, best_threshold, False)
