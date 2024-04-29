import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import pandas as pd
from bayes_opt import BayesianOptimization
import warnings
import pickle
from spyder.utils.external.pybloom_pyqt import BloomFilter
from lib.data_processing import cal_region_id
import os

warnings.filterwarnings("ignore", category=UserWarning)


# update逻辑 先通过模型+BF判断对错，然后再通过模型判断数据，插入布隆过滤器


class fpr_loss(nn.Module):
    def __init__(self, all_record, all_memory, model_size, epsilon=1e-12):
        super(fpr_loss, self).__init__()
        self.epsilon = epsilon  # 添加一个小常数以避免计算 log(0)
        self.all_record = all_record
        self.memory = all_memory - model_size
        if self.memory <= 0:
            raise ValueError('memory is zero')
    def forward(self, y_pred, y_true, val_acc):
        y_pred = torch.clamp(y_pred, self.epsilon, 1 - self.epsilon)
        bf_rate = torch.pow(2, -(self.memory / (y_pred - y_pred * y_true + self.all_record * val_acc) * torch.log(
            torch.tensor(2))))
        #
        bf_rate = torch.clamp(bf_rate, self.epsilon, 1 - self.epsilon)
        if torch.tensor(0) in y_pred - y_pred * y_true + self.all_record * val_acc:
            raise ValueError('y_pred - y_pred * y_true + self.all_record * val_acc is zero')

        bce_loss = -y_true * torch.log(y_pred) - (1 - y_true) * torch.log(1 - y_pred) - ((1 - y_true) * (
                1 - y_pred) * torch.log(1 - bf_rate))
        return torch.mean(bce_loss)


class SimpleNetwork(nn.Module):
    def __init__(self, structure, input_dim, output_dim):
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        super(SimpleNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.structure = structure
        self.layers = self._build_layers()

    def _build_layers(self):
        layers = []
        prev_layer_dim = self.input_dim
        for layer_dim in self.structure:
            layers.append(nn.Linear(prev_layer_dim, layer_dim))
            layers.append(nn.Sigmoid())  # 在隐藏层应用 Sigmoid 激活函数
            prev_layer_dim = layer_dim
        # 在输出层应用 Sigmoid 激活函数
        layers.append(nn.Linear(prev_layer_dim, self.output_dim))
        layers.append(nn.Sigmoid())  # 在输出层应用 Sigmoid 激活函数
        return nn.Sequential(*layers).to(self.device)

    def forward(self, x):
        return self.layers(x)

    def get_structure(self):
        return self.structure


# 根据阈值获得结果
def get_result(outputs, stand=0.5):
    zero = torch.zeros_like(outputs)
    one = torch.ones_like(outputs)
    # a中大于0.stand的用zero(1)替换,否则a替换,即不变
    predicted = torch.where(outputs >= stand, one, outputs)
    predicted = torch.where(outputs < stand, zero, predicted)
    return predicted.int()


def get_model_size(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size_bit = (param_size + buffer_size)
    # all_size = all_size_bit / 1024 / 1024
    # print('模型总大小为：{:.3f}MB'.format(all_size))
    return all_size_bit


def train(model, train_loader, val_loader, all_memory=5 * 1024 * 1024,
          all_record=12854455, num_epochs=100, output_acc=True):
    model_size = get_model_size(model)
    criterion = fpr_loss(all_record=all_record, all_memory=all_memory, model_size=model_size, epsilon=1e-12)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    val_FPR = torch.rand(1) * 0.1
    best_fpr = 999
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0
        total_correct = 0
        total_samples = 0
        train_false_positives = 0
        train_false_negatives = 0
        val_false_positives = 0
        val_false_negatives = 0
        train_false_data_cnt = 0
        train_true_data_cnt = 0
        validation_false_data_cnt = 0
        validation_true_data_cnt = 0
        train_acc_list = []
        train_loss_list = []
        train_FPR_list = []
        train_FNR_list = []
        val_acc_list = []
        val_loss_list = []
        val_FPR_list = []
        val_FNR_list = []

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            targets = torch.tensor(targets)
            targets = torch.tensor(targets, dtype=torch.int32)
            optimizer.zero_grad()
            outputs = model(inputs)
            # predicted = torch.round(outputs).int()
            # print("output",outputs)
            predicted = get_result(outputs)
            total_samples += targets.size(0)
            targets = torch.tensor(targets, dtype=torch.int64)

            targets_float = targets.float()
            loss = criterion(outputs, targets_float, val_FPR)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            targets = targets.int()
            predicted = predicted.int()
            total_correct += (predicted == targets).sum().item()
            train_false_positives += ((predicted == 1) & (targets == 0)).sum().item()
            train_false_negatives += ((predicted == 0) & (targets == 1)).sum().item()
            train_false_data_cnt += (targets == 0).sum().item()
            train_true_data_cnt += (targets == 1).sum().item()

        train_accuracy = total_correct / total_samples
        train_FPR = train_false_positives / train_false_data_cnt
        train_FNR = train_false_negatives / train_true_data_cnt
        train_acc_list.append(train_accuracy)
        train_loss_list.append(running_loss)
        train_FNR_list.append(train_FNR)
        train_FPR_list.append(train_FPR)

        model.eval()  # 设置模型为评估模式
        val_correct = 0
        val_samples = 0
        val_running_loss = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                targets = torch.tensor(targets)
                outputs = model(inputs)
                # predicted = torch.round(outputs).int()
                predicted = get_result(outputs)
                loss = criterion(outputs, targets.float(), val_FPR)
                val_running_loss += loss.item()
                val_samples += targets.size(0)
                targets = targets.int()
                predicted = predicted.int()
                val_correct += (predicted.int() == targets.int()).sum().item()
                val_false_positives += ((predicted == 1) & (targets == 0)).sum().item()
                val_false_negatives += ((predicted == 0) & (targets == 1)).sum().item()
                validation_false_data_cnt += (targets == 0).sum().item()
                validation_true_data_cnt += (targets == 1).sum().item()

        val_accuracy = val_correct / val_samples
        val_FPR = val_false_positives / validation_false_data_cnt
        val_FNR = val_false_negatives / validation_true_data_cnt
        val_acc_list.append(val_accuracy)
        val_loss_list.append(val_running_loss)
        val_FNR_list.append(val_FNR)
        val_FPR_list.append(val_FPR)
        if (val_FPR < best_fpr) and (val_FPR != 0):
            best_fpr = val_FPR
        if output_acc:
            print(
                f"Epoch {epoch + 1} - Loss: {running_loss:.4f} - "
                f"Train Accuracy: {train_accuracy:.4f} - Val Accuracy: {val_accuracy:.4f}")
            print(
                f"Epoch {epoch + 1} - train_FPR: {train_FPR:.4f} - train_FNR: {train_FNR:.4f} "
                f"- val_FPR: {val_FPR:.4f} - val_FNR: {val_FNR:.4f}")

    return best_fpr


def region_mapping(d, region_dict):
    region_id = cal_region_id(lat=d['lat'], lon=d['lon'])
    if region_id in region_dict:
        return region_dict[region_id]
    else:
        return np.zeros(24)


# 将时间戳处理成时间桶
def time_embedding(time_str):
    time = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
    # 生成时间的one-hot向量
    time_bucket = time.hour * 2 + time.minute // 30
    time_vec = np.eye(48)[time_bucket]
    return time_vec


# keywords embedding
# cnt=0
def keywords_embedding(keyword, word_dict):
    if keyword in word_dict:
        return word_dict[keyword]
    else:
        # cnt用于记录没找到对应的关键字的
        # cnt+=1
        # print(cnt)
        return np.zeros(300)


def insert(ck):
    time = ck['timestamp']
    keywords = ck['keywords']
    lat = ck['lat']
    lon = ck['lon']
    time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
    time = str(time.year) + str(time.month).zfill(2) + str(time.day).zfill(2) + str(time.hour).zfill(2) + str(
        time.minute).zfill(2)
    # print(time)
    region_id = str(cal_region_id(lat, lon)).zfill(8)
    try:
        keywords = keywords.replace(" ", "")
    except AttributeError:
        keywords = ''
    return time + region_id + keywords


def to_embedding(d):
    region = np.array(d['region'])
    time = np.array(d['timestamp'])
    keywords = np.array(d['keywords'])
    embedding = torch.tensor(np.concatenate((time, region, keywords)), dtype=torch.float32)
    # print(embedding.shape)
    return embedding


def validate(model, word_dict, region_dict, dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    # 逐块读取validation_set
    chunk_size = 10000
    chunk_no = 0
    threshold = 0.5
    predict_true = 0
    predict_false = 0
    for chunk in pd.read_csv('dataset/' + dataset + '/vali_data.csv', chunksize=chunk_size):
        # chunk.drop(columns='is_in',inplace=True)
        # 处理一个用于插入布隆过滤器的dataframe:data_insert
        data = chunk

        data['keywords'] = data['keywords'].str.split(' ')
        data = data.explode('keywords')
        data = data.reset_index(drop=True)
        data['keywords'] = data['keywords'].apply(str.lower)

        data_insert = pd.DataFrame()
        data_insert['insert'] = data.apply(insert, axis=1)

        # region embedding
        data['region'] = data.apply(region_mapping, axis=1, args=(region_dict,))
        data.drop(columns=['lat', 'lon'], inplace=True)

        # time embedding
        data['timestamp'] = data['timestamp'].apply(time_embedding)

        # keywords embedding
        data['keywords'] = data['keywords'].apply(keywords_embedding, args=(word_dict,))

        # 生成一个用于神经网络输入的dataframe:embedding
        embedding = pd.DataFrame()
        embedding['embedding'] = data.apply(to_embedding, axis=1)

        # 清理内存
        del data
        del chunk

        with torch.no_grad():  # 不计算梯度
            output_tensor = embedding['embedding'].apply(lambda row: model(row.to(device)).cpu().numpy())
            binary_output = np.where(output_tensor > threshold, 1, 0)
            num_ones = np.count_nonzero(binary_output)
            num_zeros = len(binary_output) - num_ones
        zero_indices = np.where(binary_output == 0)[0]
        data_insert = data_insert.iloc[zero_indices]
        csv_filename = 'insert_tweet_data.csv'

        # 检查文件是否存在，如果不存在则创建它
        if not os.path.exists(csv_filename):
            # 创建一个空的DataFrame，并保存到CSV文件中
            pd.DataFrame(columns=['insert']).to_csv(csv_filename, index=False)
        data_insert.to_csv(csv_filename, mode='a', header=not pd.read_csv(csv_filename).index.any(), index=False)
        predict_true = predict_true + num_ones
        predict_false = num_zeros + predict_false
        chunk_no += 1
        print("预测到块:", chunk_no)
        print("大于阈值的个数:", predict_true)
        print("小于等于阈值的个数:", predict_false)


def create_bloom_filter(dataset, bf_name):
    data = pd.read_csv("insert_data_" + dataset + ".csv")
    data = data['insert']
    bloom_filter = BloomFilter(capacity=2725000, error_rate=0.0006)
    for i in range(1, len(data) - 1):
        bloom_filter.add(data[i])

    with open(bf_name, 'wb') as bf_file:
        pickle.dump(bloom_filter, bf_file)

    print("布隆过滤器已保存")


def query(model, word_dict, region_dict, dataset, bloom_filter):
    chunk_size = 10000
    chunk_no = 0
    threshold = 0.5
    acc = 0
    fpr = 0
    fnr = 0
    true_sample = 0
    false_sample = 0

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    for chunk in pd.read_csv(dataset + '_data\\query_set.csv', chunksize=chunk_size):

        # 处理一个用于插入布隆过滤器的dataframe:data_insert
        data = chunk

        data['keywords'] = data['keywords'].str.split(' ')
        data = data.explode('keywords')
        data = data.reset_index(drop=True)
        data['keywords'] = data['keywords'].apply(str.lower)

        label = data['is_in']
        true_sample = (label == 1).sum()
        false_sample = (label == 0).sum()

        data_insert = pd.DataFrame()
        data_insert = data.apply(insert, axis=1)

        # region embedding
        data['region'] = data.apply(region_mapping, axis=1, args=(region_dict,))
        data.drop(columns=['lat', 'lon'], inplace=True)

        # time embedding
        data['time'] = data['time'].apply(time_embedding)

        # keywords embedding
        data['keywords'] = data['keywords'].apply(keywords_embedding, args=(word_dict,))

        # 生成一个用于神经网络输入的dataframe:embedding
        embedding = pd.DataFrame()
        embedding['embedding'] = data.apply(to_embedding, axis=1)

        # 清理内存
        del data
        del chunk

        with torch.no_grad():  # 不计算梯度
            output_tensor = embedding.apply(lambda row: model(row['embedding'].to(device)).cpu().numpy(), axis=1)
            binary_output = np.where(output_tensor > threshold, 1, 0)

        # print(len(binary_output))
        for i in range(len(binary_output)):
            if binary_output[i] == 1:
                if label[i] == 1:
                    acc = acc + 1
                elif label[i] == 0:
                    fpr = fpr + 1
            else:
                if data_insert[i] in bloom_filter:
                    if label[i] == 1:
                        acc = acc + 1
                    elif label[i] == 0:
                        fpr = fpr + 1
                else:
                    if label[i] == 1:
                        fnr = fnr + 1
                        print(i, data_insert[i])
                    elif label[i] == 0:
                        acc = acc + 1

        chunk_no += 1
        print("预测到块:", chunk_no)
        print("acc_rate:", acc)
        print("fpr:", fpr / false_sample)
        print("fnr:", fnr / true_sample)


class Bayes_Optimizer:
    def __init__(self, input_dim, output_dim, train_loader, val_loader, learning_rate=0.005,
                 hidden_units=(8, 512)):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = learning_rate
        self.hidden_units = hidden_units
        self.best_model = None

    def target_function(self, num_hidden_units):
        # 确保 num_hidden_units 是整数
        num_hidden_units = int(num_hidden_units)

        # 初始化模型
        model = SimpleNetwork([num_hidden_units], input_dim=self.input_dim, output_dim=self.output_dim)

        # 训练模型
        return train(model, train_loader=self.train_loader, val_loader=self.val_loader, num_epochs=10, output_acc=True)

    def optimize(self):
        optimizer = BayesianOptimization(
            f=self.target_function,
            pbounds={'num_hidden_units': self.hidden_units},
            random_state=42,
        )
        optimizer.maximize(n_iter=10)

        best_params = optimizer.max['params']
        best_num_hidden_units = int(best_params['num_hidden_units'])

        # 使用最佳参数重新训练模型
        best_model = SimpleNetwork([best_num_hidden_units], input_dim=self.input_dim, output_dim=self.output_dim)
        print(optimizer.max)
        return best_model
