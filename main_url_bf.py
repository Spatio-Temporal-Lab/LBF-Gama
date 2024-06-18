import numpy as np
import pandas as pd

import lib.bf_util

# run these code to generate dataset
"""
# 加载数据集
df = pd.read_csv('dataset/url.csv')

# 将标签（1）和标签（0）分别过滤出来
df_1 = df[df['url_type'] == 1]
df_0 = df[df['url_type'] == 0]

# 从标签为'b'的样本中随机抽取和's'标签相同数量的样本
df_0_sample = df_0.sample(frac=0.8, random_state=42)

# 合并得到训练集+测试集
df_train_test = pd.concat([df_1, df_0_sample])

# 剩下的标签为'b'的样本作为查询集
df_query = df_0.drop(df_0_sample.index)

df_train, df_test = train_test_split(df_train_test, test_size=0.2, random_state=42)

df_train.to_csv('dataset/url_train.csv', index=False)
df_test.to_csv('dataset/url_test.csv', index=False)
df_query.to_csv('dataset/url_query.csv', index=False)
"""

df_train = pd.read_csv('dataset/url_train.csv')
df_test = pd.read_csv('dataset/url_test.csv')
df_query = pd.read_csv('dataset/url_query.csv')

# 获取训练集中url_type为1的行的索引
id_train = df_train[df_train['url_type'] == 1].index.tolist()

# 获取测试集中url_type为1的行的索引
id_test = df_test[df_test['url_type'] == 1].index.tolist()

# 组合训练集和测试集的url_type为1的url数据
combined_data = np.concatenate((df_train.loc[id_train, 'url'].values, df_test.loc[id_test, 'url'].values), axis=0)

# 定义布隆过滤器初始大小
initial_size = 32
max_size = 512

# 循环，从32开始，每次乘以2，直到256
size = initial_size
while size <= max_size:
    bloom_size = size * 1024
    bloom_filter = lib.bf_util.create_bloom_filter(dataset=combined_data, bf_size=bloom_size)

    # 统计假阳性率
    fp = 0
    fn = 0
    total_neg = 0
    # 遍历df_query中的每一个url列来查询布隆过滤器
    for index, row in df_query.iterrows():
        url = row['url']
        true_label = row['url_type']  # 0为负例，1为正例

        if true_label == 0:
            total_neg += 1
            if url in bloom_filter:
                fp = fp + 1
        else:
            print('contain positive query')
            if url not in bloom_filter:
                fn = fn + 1
                print(f'error for url {url}')

    print(f'fpr: {fp / total_neg}')
    size *= 2