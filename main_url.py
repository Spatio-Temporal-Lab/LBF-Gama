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

# 标签编码
# df['Label'] = df['Label'].apply(lambda x: 1 if x == 's' else 0)

# 数据预处理
# def check_multicollinearity(df, threshold=0.7):
#     df = pd.DataFrame(df)  # Convert dataset to a DataFrame if needed
#     numeric_cols = df.select_dtypes(include=[np.number]).columns
#     df_numeric = df[numeric_cols]
#     corr_matrix = df_numeric.corr().abs()  # Calculate the correlation matrix
#     cols = corr_matrix.columns
#     multicollinear_features = set()
#
#     for i in range(len(cols)):
#         for j in range(i + 1, len(cols)):
#             if corr_matrix.iloc[i, j] >= threshold:
#                 multicollinear_features.add(cols[i])
#                 multicollinear_features.add(cols[j])
#     return multicollinear_features


# multicollinear_cols = check_multicollinearity(df)
# df.drop(multicollinear_cols, axis=1, inplace=True)

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
print(validate_df)
# 如果需要拆分特征和标签，可以如下进行
X = train_df.drop('url_type', axis=1).values.astype(np.float32)
y = train_df['url_type'].values.astype(np.float32)
X_query = validate_df.drop('url_type', axis=1).values.astype(np.float32)
y_query = validate_df['url_type'].values.astype(np.float32)

# # 数据预处理
# X = df.drop(columns=['Label']).values.astype(np.float32)
# y = df['Label'].values.astype(np.float32)
#
# # 数据标准化
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
all_memory = 128 * 1024  # tweet模型大小：5 * 1024 * 1024
all_record = df.size
learning_rate = 0.001
hidden_units = (8, 512)

# nas_opt = lib.network_url.Bayes_Optimizer(input_dim=input_dim, output_dim=output_dim, train_loader=train_loader,
#                                           val_loader=test_loader, true_data_count=len(df_s),
#                                           hidden_units=hidden_units, all_record=all_record, all_memory=all_memory)
# model = nas_opt.optimize()
# print("has optimized")
# lib.network_url.train(model, train_loader=train_loader, n_true=len(df_s),
#                       bf_memory=all_memory-lib.network_url.get_model_size(model), num_epochs=30,
#                       val_loader=test_loader)
# torch.save(model, 'best_url_model.pth')
#
model = lib.network_url.SimpleNetwork([8, 32, 8], input_dim=input_dim, output_dim=output_dim)
lib.network_url.train(model, train_loader=train_loader, bf_memory=all_memory - lib.network_url.get_model_size(model),
                      n_true=len(df_s), val_loader=test_loader, num_epochs=30)

# lib.network_url.train_with_fpr_1(model, train_loader=train_loader,
#                                  bf_memory=all_memory - lib.network_url.get_model_size(model),
#                                  true_data_count=len(df_s), val_loader=test_loader, num_epochs=30)

data_negative = lib.network_url.validate(model, X_train, y_train, X_test, y_test)

model.eval()

# 获得学习模型的内存大小
model_size = lib.network_url.get_model_size(model)
bloom_size = all_memory - model_size

bloom_filter = lib.network_url.create_bloom_filter(dataset=data_negative, bf_name='best_higgs_bf_3000',
                                                   bf_size=bloom_size)

# 访问布隆过滤器的 num_bits 属性
num_bits = bloom_filter.num_bits

# 将比特位转换为字节（8 bits = 1 byte）
memory_in_bytes = num_bits / 8
print("memory of bloom filter: ", memory_in_bytes)
print("memory of learned model: ", model_size)

fpr = lib.network_url.query(model, bloom_filter, X_query, y_query)