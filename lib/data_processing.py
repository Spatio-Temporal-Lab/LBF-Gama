import csv
import numpy as np
import math
from datetime import datetime
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = torch.tensor(self.data[index], dtype=torch.float32)
        y = torch.tensor(self.labels[index], dtype=torch.float32)
        return x, y


def cal_region_id(lon, lat, x_min=27, x_max=54, y_min=-120, y_max=-74, one_kilo=0.009):
    lon, lat = float(lon), float(lat)
    lonTotal = math.ceil((y_max - y_min) / one_kilo)
    if x_min <= lat <= x_max and y_min <= lon <= y_max:
        x_num = math.ceil((lat - x_min) / one_kilo)
        y_num = math.ceil((lon - y_min) / one_kilo)
        square_num = x_num * lonTotal + y_num
        # print(square_num)
        return square_num
    else:
        return None


def loading_embedding(dataset):
    # 加载区域的编码信息
    data = pd.read_csv('embedding/region_embedding_new.csv')
    data['Merged_List'] = np.array(data[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14',
                                         '15', '16', '17', '18', '19', '20', '21', '22', '23']].apply(
        lambda x: x.tolist(), axis=1))
    data = data.drop(
        columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                 '19', '20', '21', '22', '23'])
    # print(data)
    region_dict = dict(zip(data['region_id'], data['Merged_List']))
    print('region embedding ready')

    # 读取训练的关键字embedding
    word_dict = {}
    with open('embedding/' + dataset + '_keywords_embedding.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        # 遍历csv文件中的每一行并将其添加到字典中
        for row in reader:
            # 将向量字符串转换为浮点数列表
            vector_str = row[1].strip("[]").split()
            vector = [float(x) for x in vector_str]
            # 将列表形式的向量转换成 NumPy 数组的一维向量
            vector = np.array(vector, dtype=np.float32)
            # 将关键字和向量添加到字典中
            word_dict[row[0].lower()] = vector

    print("keywords embedding ready")
    return word_dict, region_dict


def loading_data(dataset, region_dict, word_dict, dataset_type):
    # 根据不同数据集读取不同位置的数据集

    file_location = ""
    if dataset == 'tweet':
        file_location_true = "dataset/tweet/" + dataset + ".csv"
        file_location_fake = "dataset/tweet/fake_data_" + dataset + ".csv"
    if dataset == "yelp":
        # file_location_true = "dataset/yelp/" + dataset + ".csv"
        file_location_true = "dataset/yelp/validate.csv"
        file_location_fake = "dataset/yelp/fake_data_" + dataset+ ".csv"

    data_true = []
    data_fake = []
    head = ["data","label"]
    data_true.append(head)
    data_fake.append(head)
    # 加载数据 真数据
    print("loading data true")
    i = 0
    with open(file_location_true, newline='', encoding='utf-8') as csvfile:
        data_true = []
        label_true = []
        data_false = []
        label_false = []
        reader = csv.reader(csvfile)
        next(reader)  # 跳过表头
        for row in reader:
            time_str = row[0]
            lon, lat = row[3], row[2]
            label = 1
            # 生成区域的embedding
            region_id = cal_region_id(lon, lat)
            if region_id in region_dict:
                space = region_dict[region_id]
            else:
                space = np.zeros(24)

            # 生成时间的one-hot向量
            time = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')

            time_bucket = time.hour * 2 + time.minute // 30
            time_vec = np.eye(48)[time_bucket]

            # 解析关键字，生成多条数据，每条数据的关键字不同
            keywords = row[1].split()
            for keyword in keywords:
                # 将输入和输出添加到数组中
                keyword = keyword.lower()
                if keyword in word_dict.keys():
                    word_vec = word_dict[keyword]
                else:
                    raise ValueError
                a = np.concatenate((time_vec, space, word_vec))
                # if int(label) == 1:
                #     data_true.append(a)
                #     label_true.append([int(label)])
                # else:
                #     data_false.append(a)
                #     label_false.append([int(label)])
                row = []
                row.append(a)
                row.append(1)
                data_true.append(row)
            i += 1
            print(i)

    print("loading data fake")
    # 加载数据 假数据
    with open(file_location_fake, newline='', encoding='utf-8') as csvfile:
        data_true = []
        label_true = []
        data_false = []
        label_false = []
        reader = csv.reader(csvfile)
        next(reader)  # 跳过表头
        for row in reader:
            time_str = row[0]
            lon, lat = row[3], row[2]
            label = 0
            # 生成区域的embedding
            region_id = cal_region_id(lon, lat)
            if region_id in region_dict:
                space = region_dict[region_id]
            else:
                space = np.zeros(24)

            # 生成时间的one-hot向量
            time = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')

            time_bucket = time.hour * 2 + time.minute // 30
            time_vec = np.eye(48)[time_bucket]

            # 解析关键字，生成多条数据，每条数据的关键字不同
            keywords = row[1].split()
            for keyword in keywords:
                # 将输入和输出添加到数组中
                keyword = keyword.lower()
                if keyword in word_dict.keys():
                    word_vec = word_dict[keyword]
                else:
                    raise ValueError
                a = np.concatenate((time_vec, space, word_vec))
                row = []
                row.append(a)
                row.append(0)
                data_fake.append(a)
    df_s = pd.DataFrame(data_true)  # 将列表数据转化为 一列
    df_b = pd.DataFrame(data_fake)  # 将列表数据转化为 一列
    return df_s,df_b

