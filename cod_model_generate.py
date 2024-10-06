import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

import lib.lgb_url
import lib.bf_util


def get_model(max_model_memory):
    df_train = pd.read_csv('Train_COD.csv')
    df_test = pd.read_csv('Test_COD.csv')

    X_train = df_train.drop(columns=['type']).values.astype(np.float32)
    y_train = df_train['type'].values.astype(np.float32)
    X_test = df_test.drop(columns=['type']).values.astype(np.float32)
    y_test = df_test['type'].values.astype(np.float32)

    train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    test_data = lgb.Dataset(X_test, label=y_test, free_raw_data=False)
    n_true = df_train[df_train['type'] == 1].shape[0] + df_test[df_test['type'] == 1].shape[0]


    best_params = None
    best_score = float('inf')

    # 定义参数空间
    num_leaves_list = range(16, 31)  # 叶子数量从2到31
    num_rounds_list = range(16, 31)  # 训练轮次从1到20
    # best_params = {'num_leaves': 65, 'num_rounds': 65}
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


get_model(200 * 1024)
