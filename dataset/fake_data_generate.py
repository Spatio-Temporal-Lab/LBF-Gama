import pandas as pd
import numpy as np
import csv
import random
from datetime import datetime, timedelta
from collections import Counter
import math

data = pd.read_csv('tweet_clean.csv')

# 获取行数
number_of_rows = data.shape[0]

# 打印行数
print(f"The number of rows in the CSV file is: {number_of_rows}")
# 构建词库 word_list为数据集全体关键字 chooose_term表示供选择的出现次数最高的9w个关键字
# 90000=30000*3 即生成条数乘以每条中关键字的期望个数(假定为3 没那么多关键字了 现实查询也可能出现重复)
word_list = data['keywords'].str.split().explode().tolist()
word_list = [word for word in word_list if not isinstance(word, float)]
# #print(len(word_list))
# unique_word_list=set(word_list)
# #print(len(unique_word_list))
# word_count = Counter(word_list)
# tf = list(word_count.items())
# max_freq = max(tf, key=lambda x: x[1])[1]
# tf = sorted(tf, key=lambda x: x[1], reverse=True)
# choose_term = tf[:200000]
# choose_term = [x[0] for x in choose_term]
# print(len(choose_term))


# #截取所需部分 如条数或指定区域
# data=data.sample(int(number_of_rows * 0.01))
# data=data.loc[(data['2']>=39.90) & (data['2']<=40.30) & (data['3']>=-75.50) & (data['3']<=-75.10)]


# 获取数据区域内最晚和最早的时间以随机生成时间戳
max_timestamp = max(data['time'])
min_timestamp = min(data['time'])
print(max_timestamp)
print(min_timestamp)

# Define the format of the date string
date_format = "%Y-%m-%d %H:%M:%S"

start_datetime = datetime.strptime(min_timestamp, date_format)
end_datetime = datetime.strptime(max_timestamp, date_format)
max_lat = 54
min_lat = 27
max_lon = -74
min_lon = -120


def random_timestamp(placeholder):
    random_datetime = start_datetime + timedelta(
        seconds=random.randint(0, int((end_datetime - start_datetime).total_seconds())))
    return datetime.utcfromtimestamp(random_datetime.timestamp()).strftime('%Y-%m-%d %H:%M:%S')


# 获取数据区域内的最大最小经纬度范围 随机生成该范围内的经纬度
def random_lon(placeholder):
    random_lon = random.uniform(min_lon, max_lon)
    return "{:.5f}".format(random_lon)


def random_lat(placeholder):
    random_lat = random.uniform(min_lat, max_lat)
    return "{:.5f}".format(random_lat)


'''
maxn=0
minn=9999
for word in word_list:
    try:
        if(maxn<len(word)): maxn=len(word)
        if(minn>len(word)): minn=len(word)
    except:
        pass
print(maxn)
print(minn)
'''


# maxn=140 minn=1


def random_keywords(placeholder):
    word_num = int(random.uniform(1, 10))  # 一条数据包含1-7个关键字
    random_wordlist = random.sample(word_list, word_num)
    return ' '.join(random_wordlist)


def timestamp2onehot(time_str):
    time = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')

    time_bucket = time.hour * 2 + time.minute // 30
    time_vec = np.eye(48)[time_bucket]
    return time_vec


def cal_region_id(lon, lat, x_min=27, x_max=54, y_min=-120, y_max=-74, oneKilo=0.009):
    lat = float(lat)
    lon = float(lon)
    latTotal = math.ceil((x_max - x_min) / oneKilo)  # 纬度方向有多少个格子
    square_num = 0
    if x_min <= lat <= x_max and lon >= y_min and lon <= y_max:
        x_num = math.ceil((lat - x_min) / oneKilo)
        y_num = math.ceil((lon - y_min) / oneKilo)
        # print("x_num ", x_num, "y_num ", y_num)

        if x_num == 0:
            if y_num == 0:
                square_num = 1
            else:
                square_num = (y_num - 1) * latTotal + 1
        else:
            square_num = (y_num - 1) * latTotal + x_num
    return square_num


# 生成负样本
fake_data = pd.DataFrame()
fake_data['time'] = data['time'].apply(random_timestamp)
fake_data['keywords'] = data['keywords'].apply(random_keywords)
fake_data['lat'] = data['lat'].apply(random_lat)
fake_data['lon'] = data['lon'].apply(random_lon)

print(fake_data)

# 去重
data['time']=data['time'].apply(timestamp2onehot)
data['region_id'] = data.apply(cal_region_id(data['lon']))
mask = ~fake_data[['time', 'keywords', 'lat', 'lon']].isin(data[['time', 'keywords', 'lat', 'lon']])
fake_data = fake_data[mask]

fake_data['is_in'] = 0

print(fake_data)
fake_data.to_csv('fake_data_query_tweet.csv', index=False)
