import os
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

import lib.network_url


# 序列化模型对象


class HIGGSDataset(Dataset):
    def __init__(self, X_, y_):
        self.X = torch.tensor(X_)
        self.y = torch.tensor(y_)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 加载数据集
df = pd.read_csv('dataset/train.csv')

# 将's'标签（1）和'b'标签（0）分别过滤出来
df_s = df[df['url_type'] == 1]
df_b = df[df['url_type'] == 0]
# 取出标签为's'的所有样本数
s_count = len(df_s)

# 从标签为'b'的样本中随机抽取和's'标签相同数量的样本
# df_b_sample = df_b.sample(n=s_count, random_state=42)
df_b_sample = df_b.sample(frac=0.8, random_state=42)

# 合并得到训练集（1:1比例）
train_df = pd.concat([df_s, df_b_sample])

# 剩下的标签为'b'的样本作为测试集
validate_df = df_b.drop(df_b_sample.index)
print(validate_df)
# 如果需要拆分特征和标签，可以如下进行
X = train_df.drop('url_type', axis=1).values.astype(np.float32)
y = train_df['url_type'].values.astype(np.float32)
X_query = validate_df.drop('url_type', axis=1).values.astype(np.float32)
y_query = validate_df['url_type'].values.astype(np.float32)

# # 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_query = scaler.fit_transform(X_query)
#
# # 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

# 设置参数
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 63,
    'learning_rate': 0.1,
    'feature_fraction': 1,
}
all_memory = 64 * 1024  # tweet模型大小：5 * 1024 * 1024

num_round = 100
bst = lgb.train(params, train_data, num_round, valid_sets=[test_data])
bst.save_model('bst_model.txt')
model_size = os.path.getsize('bst_model.txt')
print("模型在内存中所占用的大小（字节）:", model_size)
with open('bst_model.pkl', 'wb') as f:
    pickle.dump(bst, f)

threshold = 0.99
data_negative = lib.network_url.lightgbm_validate(bst, X_train, y_train, X_test, y_test, threshold)
model_size = lib.network_url.bst_get_model_size(bst)
bloom_size = all_memory - model_size

bloom_filter = lib.network_url.create_bloom_filter(dataset=data_negative, bf_name='best_higgs_bf_3000',
                                                   bf_size=bloom_size)
# with open('best_higgs_bf_3000', 'rb') as bf_file:
#     bloom_filter = pickle.load(bf_file)

# 访问布隆过滤器的 num_bits 属性
num_bits = bloom_filter.num_bits

# 将比特位转换为字节（8 bits = 1 byte）
memory_in_bytes = num_bits / 8
print("memory of bloom filter: ", memory_in_bytes)
print("memory of learned model: ", model_size)

fpr = lib.network_url.bst_query(bst, bloom_filter, X_query, y_query, threshold)

