import lightgbm as lgb
import pandas as pd
import lib.network
import lib.data_processing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np
import pickle

data = pd.read_csv('dataset/yelp/query_data.csv')
word_dict, region_dict = lib.data_processing.loading_embedding("yelp")

data['keywords'] = data['keywords'].str.split(' ')
data = data.explode('keywords')
data = data.reset_index(drop=True)
data['keywords'] = data['keywords'].astype(str)

data['keywords'] = data['keywords'].apply(str.lower)

data_insert = pd.DataFrame()
data_insert['insert'] = data.apply(lib.network.insert, axis=1)

# region embedding
data['region'] = data.apply(lib.network.region_mapping, axis=1, args=(region_dict,))
data.drop(columns=['lat', 'lon'], inplace=True)

# time embedding
data['timestamp'] = data['timestamp'].apply(lib.network.time_embedding)

# keywords embedding
data['keywords'] = data['keywords'].apply(lib.network.keywords_embedding, args=(word_dict,))
print(data.head())
print(data)
# 生成一个用于神经网络输入的dataframe:embedding
embedding = pd.DataFrame()
embedding['embedding'] = data.apply(lib.network.to_embedding, axis=1)
print(embedding)
y = data['is_in']
# 清理内存
del data

X = pd.DataFrame(embedding['embedding'].apply(pd.Series))
print(X)

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.values.astype(np.float32)
X_test = X_test.values.astype(np.float32)
y_train = y_train.values.astype(np.float32)
y_test = y_test.values.astype(np.float32)
# 4. 创建 LightGBM 数据集
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# 5. 设置参数
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_error',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

# 6. 训练模型
gbm = lgb.train(params,
                train_data,
                num_boost_round=10,
                valid_sets=[train_data, test_data])

# 7. 模型预测
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# 将概率转换为类别（0或1）
y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred]

# 8. 评估模型
accuracy = accuracy_score(y_test, y_pred_binary)
print(f'Binary Classification Accuracy: {accuracy}')
with open('lgb_model.pkl', 'wb') as f:
    pickle.dump(gbm, f)
# 如果是分类任务，可以计算准确率
# accuracy = accuracy_score(y_test, (y_pred > 0.5).astype(int))
# print(f'Accuracy: {accuracy}')

