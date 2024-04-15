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


def cal_region_id(lon,lat,x_min=27,x_max=54,y_min=-120,y_max=-74,oneKilo=0.009):
        
        lat=float(lat)
        lon=float(lon)
        latTotal = math.ceil((x_max-x_min) / oneKilo)  #纬度方向有多少个格子
        lonTotal = math.ceil((y_max-y_min) / oneKilo)  #经度方向有多少个格子
        square_num = 0
        if lat >= x_min and lat <= x_max and lon >= y_min and lon <= y_max:
                x_num = math.ceil((lat - x_min) / oneKilo)
                y_num = math.ceil((lon - y_min) / oneKilo)
                #print("x_num ", x_num, "y_num ", y_num)
                
                if (x_num == 0):
                    if (y_num == 0):
                        square_num = 1
                    else:
                        square_num = (y_num - 1) * latTotal + 1
                else:
                    square_num = (y_num - 1) * latTotal + x_num
        return square_num

def loading_embedding(dataset):
    

    #加载区域的编码信息
    data=pd.read_csv('embedding\\region_embedding.csv')
    data['Merged_List'] = np.array(data[['0', '1', '2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23']].apply(lambda x: x.tolist(), axis=1))
    data = data.drop(columns=['0', '1', '2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23'])
    #print(data)
    region_dict = dict(zip(data['region_id'], data['Merged_List']))
    print('region embedding ready')

    # 读取训练的关键字embedding
    word_dict = {}
    with open('embedding\\all_keywords_embedding_'+dataset+'.csv', newline='', encoding='utf-8') as csvfile:
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
    return word_dict,region_dict



def loading_data(dataset,region_dict,word_dict,dataset_type):
    #根据不同数据集读取不同位置的数据集
    if dataset=='tweet':
        file_location="dataset\\tweet\\"+dataset_type+"_set.csv" 
    if dataset=="yelp":
        file_location="dataset\\yelp\\"+dataset_type+"_set.csv"

    #加载数据
    with open(file_location, newline='',encoding='utf-8') as csvfile:
        data_true=[]
        label_true=[]
        data_false=[]
        label_false=[]
        reader = csv.reader(csvfile)
        next(reader)      #跳过表头
        for row in reader:
            time_str, label = row[0], row[4]
            lon,lat=row[2],row[3]
            #print(lat,lon)
            #print(time_str,label,lat,lon)
            # 生成区域的embedding
            region_id=cal_region_id(lon,lat)
            if(region_id in region_dict):
                space=region_dict[region_id]
            else:
                space=np.zeros(24)

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
                #if(1):
                    # print("-----------------------")
                    # print(len(time_vec),len(space_vec),len(word_vec))
                a = np.concatenate((time_vec, space, word_vec))
                #print(len(a))
                #将真假数据区分开来
                if int(label) == 1:
                    data_true.append(a)
                    label_true.append([int(label)])
                else:
                    data_false.append((a))
                    label_false.append([int(label)])

    if dataset_type=="train":


        #Todo 修改validation占比 并且让train 和 validation使用不同的正样本
        print(f"{len(data_true)} true data for train model")
        print(f"{len(data_false)} false data for train and validate model")
        #train_false_data, test_false_data, train_false_labels, test_false_labels = train_test_split(data_false, labels_false, test_size=0.2, random_state=42)
        # Split the data into training (80%) and temporary (20%)
        train_false_data, temp_false_data, train_false_labels, temp_false_labels = train_test_split(data_false, label_false, test_size=0.2, random_state=42)

        data_train = np.concatenate((data_true, train_false_data), axis=0) #拼接训练真假数据
        label_train = np.concatenate((label_true, train_false_labels), axis=0)

        train_dataset = CustomDataset(data_train, label_train)
        train_false_data_cnt = np.count_nonzero(label_train == 0)  # train中负样本总数
        train_true_data_cnt = label_train.size - train_false_data_cnt  # train中正样本总数
        print(f"{train_false_data_cnt} false data for train model")
        print(f"{train_true_data_cnt} true data for train model")



        #用于模型验证的数据集
        data_validation = np.concatenate((data_true, temp_false_data), axis=0) #拼接val真假数据
        label_validation = np.concatenate((label_true, temp_false_labels), axis=0)
        val_dataset = CustomDataset(data_validation, label_validation)
        validation_false_data_cnt = np.count_nonzero(label_validation == 0)  # validation中负样本总数
        validation_true_data_cnt = label_validation.size - validation_false_data_cnt  # validation中正样本总数

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    return train_loader,val_loader