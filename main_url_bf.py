import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

import lib.network_url


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
df_b_sample = df_b.sample(n=s_count, random_state=42)

# 合并得到训练集（1:1比例）
train_df = pd.concat([df_s, df_b_sample])

# 剩下的标签为'b'的样本作为测试集
validate_df = df_b.drop(df_b_sample.index)

# 如果需要拆分特征和标签，可以如下进行
X = train_df.drop('url_type', axis=1).values.astype(np.float32)
y = train_df['url_type'].values.astype(np.float32)
X_query = validate_df.drop('url_type', axis=1).values.astype(np.float32)
y_query = validate_df['url_type'].values.astype(np.float32)

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_query = scaler.fit_transform(X_query)
#
# # 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_dataset = HIGGSDataset(X_train, y_train)
test_dataset = HIGGSDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

input_dim = X_train.shape[1]
print('input_dim = ', input_dim)
print('test dataset size = ', len(test_dataset))
output_dim = 1
all_memory = 64 * 1024  # tweet模型大小：5 * 1024 * 1024
all_record = df.size
learning_rate = 0.001
hidden_units = (8, 512)
bloom_size = all_memory

indices_train = [i for i in range(len(X_train)) if y_train[i] == 1]
indices_test = [i for i in range(len(X_test)) if y_test[i] == 1]

bloom_filter = lib.network_url.create_bloom_filter(
    dataset=np.concatenate((X_train[indices_train], X_test[indices_test]), axis=0), bf_name='best_higgs_bf_3000',
    bf_size=bloom_size)

# 访问布隆过滤器的 num_bits 属性
num_bits = bloom_filter.num_bits

# 将比特位转换为字节（8 bits = 1 byte）
memory_in_bytes = num_bits / 8
print("memory of bloom filter: ", memory_in_bytes)

fn = 0
fp = 0

total = len(X_query)
print(f"query count = {total}")

for i in range(total):
    input_data = X_query[i]
    true_label = y_query[i]

    if input_data in bloom_filter:
        if true_label == 0:
            fp = fp + 1
    else:
        if true_label == 1:
            fn = fn + 1

print(f"fp: {fp}")
print(f"total: {total}")
print(f"fpr: {float(fp) / total}")
print(f"fnr: {float(fn) / total}")