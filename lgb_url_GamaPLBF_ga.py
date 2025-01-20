import copy
import time
import lightgbm as lgb
import numpy as np
import pandas as pd
import pygad

import lib.bf_util
import lib.lgb_url
from plbf.FastPLBF_M import FastPLBF_M

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

positive_items = np.concatenate((X_train[y_train == 1], X_test[y_test == 1]), axis=0)
negative_items = np.concatenate((X_train[y_train == 0], X_test[y_test == 0]), axis=0)
positive_samples = X_sample[y_sample == 1]
negative_samples = X_sample[y_sample == 0]

positive_train_urls = df_train[df_train['url_type'] == 1]['url']
positive_test_urls = df_test[df_test['url_type'] == 1]['url']
positive_urls = pd.concat([positive_train_urls, positive_test_urls])
positive_urls_list = positive_urls.tolist()

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

for size in range(256 * 1024, 1280 * 1024 + 1, 256 * 1024):
    model_sizes = []
    best_fpr = 1.0
    best_plbf = None
    best_epoch = 0
    epoch_max = 200

    start_time = time.perf_counter_ns()
    cur_epoch = 0

    bst = lgb.Booster(params=params, train_set=train_data)

    fpr_map = dict()

    # 更新 objective 函数，使其适应遗传算法
    def objective(epoch):
        global cur_epoch, epoch_max
        epoch = int(epoch)
        if epoch > epoch_max:
            return -1.0

        if fpr_map.__contains__(epoch):
            return fpr_map[epoch]

        while cur_epoch < epoch:
            cur_epoch += 1
            bst.update(train_data)
            size_temp = lib.lgb_url.lgb_get_model_size(bst)
            if size_temp > size:
                epoch_max = cur_epoch - 1
                return -1.0
            model_sizes.append(size_temp)

        model_size = model_sizes[epoch - 1]  # Model size at `num_iteration`
        bf_bytes = size - model_size

        pos_scores = bst.predict(positive_samples, num_iteration=epoch).tolist()
        neg_scores = bst.predict(negative_samples, num_iteration=epoch).tolist()
        plbf = FastPLBF_M(positive_urls_list, pos_scores, neg_scores, bf_bytes * 8.0, 50, 5)
        cur_fpr = plbf.get_fpr()

        fpr_map[epoch] = -cur_fpr
        global best_fpr, best_plbf, best_epoch
        if cur_fpr < best_fpr:
            best_plbf = copy.deepcopy(plbf)
            best_fpr = cur_fpr
            best_epoch = epoch
        return -cur_fpr

    # 使用 pygad 进行优化
    def optimize_epochs():
        # 更新 fitness_func，接受 3 个参数
        def fitness_func(ga_instance, solution, solution_idx):
            epoch = int(solution[0])
            return objective(epoch)

        # 初始化遗传算法
        ga = pygad.GA(
            num_generations=50,  # 代数
            num_parents_mating=10,  # 每代父母数量
            sol_per_pop=20,  # 每代个体数量
            num_genes=1,  # 基因数目（这里只优化 epoch）
            fitness_func=fitness_func,
            gene_type=int,  # 基因类型
            gene_space={'low': 2, 'high': 200},  # 设置优化范围
            parent_selection_type="tournament",  # 父代选择方式
            crossover_type="uniform",  # 交叉方式
            crossover_probability=0.6,  # 交叉概率
            mutation_type="random",  # 变异方式
            mutation_probability=0.1,  # 变异概率
            random_seed=42
        )

        # 运行遗传算法
        ga.run()

    optimize_epochs()
    best_bst = lgb.Booster(model_str=bst.model_to_string(num_iteration=best_epoch))

    model_size = lib.lgb_url.lgb_get_model_size(best_bst)
    print("模型在内存中所占用的大小（字节）:", model_size)
    print(f"best epoch:", best_epoch)

    pos_scores = best_bst.predict(positive_items).tolist()

    fp_cnt = 0
    query_negative = X_query
    query_neg_keys = query_urls
    query_neg_scores = best_bst.predict(X_query)
    total = len(query_negative)

    best_plbf.insert_keys(positive_urls_list, pos_scores)
    end_time = time.perf_counter_ns()
    print(f'use {(end_time - start_time) / 1000000}ms')
    for key, score in zip(query_neg_keys, query_neg_scores):
        if best_plbf.contains(key, score):
            fp_cnt += 1
    print(f"fpr: {float(fp_cnt) / total}")
