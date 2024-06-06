import pickle

import numpy as np
from pybloom_live import BloomFilter

from lib import bf_util


def lgb_get_model_size(lgb_model):
    # 使用 pickle 序列化模型
    model_str = pickle.dumps(lgb_model)
    # 计算序列化后的字节数
    model_size = len(model_str)
    return model_size


def lgb_validate(model, X_train, y_train, train_urls, X_test, y_test, test_urls, threshold):
    prediction_results = model.predict(X_train)
    binary_predictions = (prediction_results > threshold).astype(int)
    indices_train = [i for i in range(len(X_train)) if binary_predictions[i] == 0 and y_train[i] == 1]

    # 对测试集进行预测
    prediction_results = model.predict(X_test)
    binary_predictions = (prediction_results > threshold).astype(int)
    indices_test = [i for i in range(len(X_test)) if binary_predictions[i] == 0 and y_test[i] == 1]

    # 提取对应的url
    urls_train = train_urls.iloc[indices_train].values
    urls_test = test_urls.iloc[indices_test].values

    # 返回合并后的url数组
    return np.concatenate((urls_train, urls_test), axis=0)


def create_bloom_filter(dataset, bf_size):
    n_items = len(dataset)
    print('n_items = ', n_items)
    print('bf_size = ', bf_size)

    # # 创建布隆过滤器
    bloom_filter = BloomFilter(capacity=max(1, n_items), error_rate=bf_util.get_fpr(n_items, bf_size))
    for data in dataset:
        bloom_filter.add(data)

    return bloom_filter


def lgb_query(model, bloom_filter, X_query, y_query, query_urls, threshold, draw):
    fn = 0
    fp = 0
    cnt_ml = 0
    cnt_bf = 0
    total = len(X_query)
    print(f"query count = {total}")

    prediction_results = model.predict(X_query)

    if draw:
        import matplotlib.pyplot as plt
        bins = np.arange(0, 1.1, 0.1)  # 生成 [0, 0.1, 0.2, ..., 1.0] 的区间
        plt.hist(prediction_results, bins=bins, edgecolor='black', alpha=0.7)
        plt.xlabel('Prediction value')
        plt.ylabel('Frequency')
        plt.title('Histogram of All Predictions')
        plt.xticks(bins)
        plt.grid(axis='y', alpha=0.75)
        plt.show()

    binary_predictions = (prediction_results > threshold).astype(int)

    print(prediction_results)
    print(binary_predictions)

    for i in range(total):
        true_label = y_query[i]
        url = query_urls[i]
        prediction = binary_predictions[i]
        if prediction == 1:
            if true_label == 0:
                fp = fp + 1
                cnt_ml = cnt_ml + 1
        else:
            if url in bloom_filter:
                if true_label == 0:
                    fp = fp + 1
                    cnt_bf = cnt_bf + 1
            else:
                if true_label == 1:
                    fn = fn + 1

    print(f"fp: {fp}")
    print(f"total: {total}")
    print(f"fpr: {float(fp) / total}")
    print(f"fnr: {float(fn) / total}")
    print(f"cnt_ml: {cnt_ml}")
    print(f"cnt_bf: {cnt_bf}")
    return float(fp) / total
