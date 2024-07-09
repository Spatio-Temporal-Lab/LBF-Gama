import numpy as np
import pandas as pd
from datetime import datetime
import math
import lib.bf_util

# 加载数据集
data_train = pd.read_csv('dataset/tweet/tweet_train.csv')
data_test = pd.read_csv('dataset/tweet/tweet_test.csv')
data_query = pd.read_csv('dataset/tweet/tweet_query.csv')



def cal_region_id(lon, lat, x_min=27, x_max=54, y_min=-120, y_max=-74, one_kilo=0.009):
    lon, lat = float(lon), float(lat)
    lonTotal = math.ceil((y_max - y_min) / one_kilo)
    if x_min <= lat <= x_max and y_min <= lon <= y_max:
        x_num = math.ceil((lat - x_min) / one_kilo)
        y_num = math.ceil((lon - y_min) / one_kilo)
        square_num = x_num * lonTotal + y_num
        return square_num
    else:
        return None


def insert(ck):
    time = ck['timestamp']
    keywords = ck['keywords']
    lat = ck['lat']
    lon = ck['lon']
    time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
    time_bucket = time.hour * 2 + time.minute // 30
    time = str(time.year) + str(time.month).zfill(2) + str(time.day).zfill(2) + str(time_bucket).zfill(2)
    region_id = str(cal_region_id(lat=lat, lon=lon)).zfill(8)
    try:
        keywords = keywords.replace(" ", "")
    except AttributeError:
        keywords = ''
    ck['insert'] = time + region_id + keywords
    return ck['insert']


def yelp_insert(data_train):
    data_train['keywords'] = data_train['keywords'].str.split(' ')
    data_train = data_train.explode('keywords')
    data_train = data_train.reset_index(drop=True)
    data_train['keywords'] = data_train['keywords'].astype(str)
    data_train['keywords'] = data_train['keywords'].apply(str.lower)
    data_train = data_train[data_train['is_in'] == 1]

    data_train['insert'] = data_train.apply(insert, axis=1)
    data = data_train[['insert', 'is_in']]
    return data


def query_insert(data_train):
    data_train['keywords'] = data_train['keywords'].str.split(' ')
    data_train = data_train.explode('keywords')
    data_train = data_train.reset_index(drop=True)
    data_train['keywords'] = data_train['keywords'].astype(str)
    data_train['keywords'] = data_train['keywords'].apply(str.lower)
    data_train['insert'] = data_train.apply(insert, axis=1)
    data = data_train[['insert', 'is_in']]
    return data


insert_train = yelp_insert(data_train)
insert_test = yelp_insert(data_test)
insert_query = query_insert(data_query)
combined_data = np.concatenate([insert_train, insert_test], axis=0)
print(combined_data)

for size in range(64 * 1024, 320 * 1024 + 1, 64 * 1024):
    bloom_filter = lib.bf_util.create_bloom_filter(dataset=combined_data, bf_size=size)

    # 统计假阳性率
    fp = 0
    fn = 0
    total_neg = 0
    # 遍历df_query中的每一个url列来查询布隆过滤器
    for index, row in insert_query.iterrows():
        url = row['insert']
        true_label = row['is_in']  # 0为负例，1为正例

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
