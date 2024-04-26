import pandas as pd
import math
from datetime import datetime

data = pd.read_csv('tweet.csv')
fake_data = pd.read_csv('fake_data_tweet.csv')

print(fake_data)
fake_data_check = fake_data


def timestamp2zerohot(time_str):
    time = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
    time_bucket = time.hour * 2 + time.minute // 30
    return time_bucket


def cal_region_id(row, x_min=27, x_max=54, y_min=-120, y_max=-74, oneKilo=0.009):
    lon, lat = float(row['lon']), float(row['lat'])
    lonTotal = math.ceil((y_max - y_min) / oneKilo)
    if x_min <= lat <= x_max and y_min <= lon <= y_max:
        x_num = math.ceil((lat - x_min) / oneKilo)
        y_num = math.ceil((lon - y_min) / oneKilo)
        square_num = x_num * lonTotal + y_num
        # print(square_num)
        return square_num
    else:
        return None


fake_data_check['region_id'] = fake_data_check.apply(cal_region_id, axis=1)
fake_data_check.drop(['lat', 'lon'], axis=1, inplace=True)
fake_data_check['timeslot'] = fake_data_check['timestamp'].apply(timestamp2zerohot)
fake_data_check.drop(['timestamp'], axis=1, inplace=True)
print(fake_data_check)

data['region_id'] = data.apply(cal_region_id, axis=1)
data.drop(['lat', 'lon'], axis=1, inplace=True)
data['timeslot'] = data['timestamp'].apply(timestamp2zerohot)
data.drop(['timestamp'], axis=1, inplace=True)
print(data)

fake_data_check['keywords'] = data['keywords'].str.split(' ')
fake_data_check = fake_data_check.explode('keywords')

data['keywords'] = data['keywords'].str.split(' ')
data = data.explode('keywords')

mask = ~fake_data_check[['keywords', 'region_id', 'timeslot']].isin(data[['keywords', 'region_id', 'timeslot']])
fake_data_check = fake_data_check[mask]
print(fake_data_check)
fake_data['is_in'] = 0

fake_data.to_csv("fake_data_checked.csv", index=False)
