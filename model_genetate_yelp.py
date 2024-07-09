import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
import lib.data_processing
import lib.network
import lib.lgb_url
import lib.bf_util


def get_model(max_model_memory):
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
        # print(embedding)
        y = data_train['is_in']
        del data_train
        X = pd.DataFrame(embedding['embedding'].apply(pd.Series))
        # print(X)
        return X, y, insert

    X_train, y_train, train_insert = yelp_embedding(data_train, word_dict=word_dict, region_dict=region_dict)
    X_test, y_test, test_insert = yelp_embedding(data_test, word_dict=word_dict, region_dict=region_dict)
    X_train = X_train.values.astype(np.float32)
    X_test = X_test.values.astype(np.float32)
    y_train = y_train.values.astype(np.float32)
    y_test = y_test.values.astype(np.float32)
    # 4. 创建 LightGBM 数据集
    train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data, free_raw_data=False)

    n_true = data_train[data_train['is_in'] == 1].shape[0] + data_test[data_test['is_in'] == 1].shape[0]

    best_params = None
    best_score = float('inf')

    # 定义参数空间
    num_leaves_list = range(2, 32)  # 叶子数量从2到31
    num_rounds_list = range(1, 21)  # 训练轮次从1到20

    # 循环遍历参数空间
    for num_leaves in num_leaves_list:
        for num_rounds in num_rounds_list:
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'num_leaves': num_leaves,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
            }

            model = lgb.train(params, train_data, num_boost_round=num_rounds, valid_sets=[test_data])
            print(
                f'num_leaves, num_rounds, memory = {num_leaves}, {num_rounds}, {lib.lgb_url.lgb_get_model_size(model)}')

            if lib.lgb_url.lgb_get_model_size(model) >= max_model_memory:
                break

            # 在验证集上评估
            valid_pred = model.predict(X_test)
            logloss_ = log_loss(y_test, valid_pred)

            # 记录最佳参数和得分
            if logloss_ < best_score:
                best_score = logloss_
                best_params = {'num_leaves': num_leaves, 'num_rounds': num_rounds}

    # 设置参数
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': best_params['num_leaves'],
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
    }
    bst = lgb.train(params, train_data, best_params['num_rounds'], valid_sets=[test_data])

    print('best num_leaves = ', best_params['num_leaves'])
    print('best num_rounds = ', best_params['num_rounds'])
    print('true data size = ', n_true)

    # 初始化变量
    model_size = lib.lgb_url.lgb_get_model_size(bst)
    print("模型在内存中所占用的大小（字节）:", model_size)
    bst.save_model('best_bst_' + str(max_model_memory))
    return bst


get_model(20 * 1024)
