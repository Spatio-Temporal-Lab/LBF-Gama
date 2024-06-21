from lib.data_processing import *
import pickle
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import lib.lgb_url

dataset = 'yelp'
word_dict, region_dict = loading_embedding(dataset)
# 将's'标签（1）和'b'标签（0）分别过滤出来
df_s, df_b = loading_data(word_dict=word_dict, region_dict=region_dict, dataset=dataset,
                                        dataset_type="train")



# 取出标签为's'的所有样本数
s_count = len(df_s)
print(s_count)

# 从标签为'b'的样本中随机抽取和's'标签相同数量的样本
# df_b_sample = df_b.sample(n=s_count, random_state=42)
df_b_sample = df_b.sample(frac=0.8, random_state=42)

# 合并得到训练集（1:1比例）
train_df = pd.concat([df_s, df_b_sample])
print(train_df)
# 剩下的标签为'b'的样本作为测试集
validate_df = df_b.drop(df_b_sample.index)
print(validate_df)

# # 如果需要拆分特征和标签，可以如下进行
X = train_df.drop('label', axis=1).values.astype(np.float32)
y = train_df['label'].values.astype(np.float32)
X_query = validate_df.drop('label', axis=1).values.astype(np.float32)
y_query = validate_df['label'].values.astype(np.float32)


# # 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_query = scaler.fit_transform(X_query)
#
# # 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)
print("has split")

# 设置参数
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.1,
    'feature_fraction': 1,
}
all_memory = 5 * 1024 * 1024  #yelp大小 64 * 1024  # tweet模型大小：5 * 1024 * 1024

num_round = 10
bst = lgb.train(params, train_data, num_round, valid_sets=[test_data])

model_size = lib.lgb_url.lgb_get_model_size(bst)
print("模型在内存中所占用的大小（字节）:", model_size)


threshold = 0.5
data_negative = lib.lgb_url.lgb_validate(bst, X_train, y_train, X_test, y_test, threshold)
bloom_size = all_memory - model_size

bloom_filter = lib.lgb_url.create_bloom_filter(dataset=data_negative, bf_name='best_tweet_bf',
                                               bf_size=bloom_size)

# 访问布隆过滤器的 num_bits 属性
num_bits = bloom_filter.num_bits

# 将比特位转换为字节（8 bits = 1 byte）
memory_in_bytes = num_bits / 8
print("memory of bloom filter: ", memory_in_bytes)
print("memory of learned model: ", model_size)

fpr = lib.lgb_url.lgb_query(bst, bloom_filter, X_query, y_query, threshold, False)