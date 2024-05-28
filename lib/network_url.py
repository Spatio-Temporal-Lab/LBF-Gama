import math
import pickle
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from bayes_opt import BayesianOptimization
from pybloom_live import BloomFilter
from torch.utils.data import DataLoader, TensorDataset

from lib import bf_util

warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


class fpr_loss(nn.Module):
    def __init__(self, all_record, all_memory, model_size, epsilon=1e-8):
        super(fpr_loss, self).__init__()
        self.epsilon = epsilon  # 添加一个小常数以避免计算 log(0)
        self.all_record = all_record
        self.memory = all_memory - model_size
        if self.memory <= 0:
            raise ValueError('memory is zero')

    def forward(self, y_pred, y_true, val_fnr):
        y_pred = torch.sigmoid(y_pred)

        denominator = y_pred - y_pred * y_true + self.all_record * val_fnr
        denominator = torch.clamp(denominator, min=self.epsilon)  # 设定一个非常小的正数作为下限
        bf_rate = torch.pow(2, -(self.memory / denominator * torch.log(torch.tensor(2.0))))

        if torch.any(torch.isnan(bf_rate)):
            raise ValueError('bf_rate contains NaN values')
            #
        bf_rate = torch.clamp(bf_rate, min=self.epsilon, max=1 - self.epsilon)

        bce_loss = -(1 - y_true) * torch.log(1 - y_pred)
        if torch.any(torch.isnan(bce_loss)):
            raise ValueError('bce_loss contains NaN values before sub')
        if y_true.any():  # 如果存在负样本
            # 只对负样本添加bf_rate相关的损失项
            negative_sample_loss = (1 - y_pred) * torch.log(1 - bf_rate)
            bce_loss -= y_true * negative_sample_loss * 0.6
        if torch.any(torch.isnan(bce_loss)):
            raise ValueError('bce_loss contains NaN values after sub')
        return torch.mean(bce_loss)


class fpr_loss_1(nn.Module):
    def __init__(self, true_data_count, bf_memory, epsilon=1e-8):
        super(fpr_loss_1, self).__init__()
        self.epsilon = epsilon  # 添加一个小常数以避免计算 log(0)
        self.true_data_count = true_data_count
        self.bf_memory = bf_memory
        if self.bf_memory <= 0:
            raise ValueError('memory is zero')

    def forward(self, y_pred, y_true, val_fnr):
        y_pred = torch.sigmoid(y_pred)

        denominator = y_pred - y_pred * y_true + self.true_data_count * val_fnr
        denominator = torch.clamp(denominator, min=self.epsilon)  # 设定一个非常小的正数作为下限
        bf_rate = torch.pow(2, -(self.bf_memory / denominator * torch.log(torch.tensor(2.0))))

        if torch.any(torch.isnan(bf_rate)):
            raise ValueError('bf_rate contains NaN values')

        bf_rate = torch.clamp(bf_rate, min=self.epsilon, max=1 - self.epsilon)
        y_pred_adjusted = torch.where(y_pred < 0.5, y_pred + bf_rate / 2, y_pred)
        bce_loss = - (y_true * torch.log(y_pred_adjusted) + (1 - y_true) * torch.log(1 - y_pred_adjusted))
        return torch.mean(bce_loss)


class SimpleNetwork(nn.Module):
    def __init__(self, structure, input_dim, output_dim):
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
            layers.append(nn.ReLU())  # 在隐藏层应用 Sigmoid 激活函数
            prev_layer_dim = layer_dim
        # 在输出层应用 Sigmoid 激活函数
        layers.append(nn.Linear(prev_layer_dim, self.output_dim))
        layers.append(nn.Sigmoid())  # 在输出层应用 Sigmoid 激活函数
        return nn.Sequential(*layers).to(device)

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
    return all_size_bit


def train(model, bf_memory, n_true, train_loader, val_loader, num_epochs=100, output_acc=True):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_fpr = 999
    best_fnr = 1.0
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

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.view(-1, 1).to(device)
            # targets = torch.tensor(targets)
            targets = torch.tensor(targets, dtype=torch.int32)
            optimizer.zero_grad()
            outputs = model(inputs)
            predicted = get_result(outputs)
            total_samples += targets.size(0)
            targets = torch.tensor(targets, dtype=torch.int64)

            targets_float = targets.float()
            loss = criterion(outputs, targets_float)  # , val_FNR
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

        model.eval()  # 设置模型为评估模式
        val_correct = 0
        val_samples = 0
        val_running_loss = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.view(-1, 1).to(device)
                targets = torch.tensor(targets)
                outputs = model(inputs)
                predicted = get_result(outputs)
                loss = criterion(outputs, targets.float())  # , val_FNR
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
        if (val_FPR < best_fpr) and (val_FPR != 0):
            best_fpr = val_FPR
            best_fnr = val_FNR
        if output_acc:
            print(
                f"Epoch {epoch + 1} - Loss: {running_loss:.4f} - "
                f"Train Accuracy: {train_accuracy:.4f} - Val Accuracy: {val_accuracy:.4f}")
            print(
                f"Epoch {epoch + 1} - train_FPR: {train_FPR:.4f} - train_FNR: {train_FNR:.4f} "
                f"- val_FPR: {val_FPR:.4f} - val_FNR: {val_FNR:.4f}")

    return 1 - (best_fpr + (1 - best_fpr) * bf_util.get_fpr(n_items=best_fnr*n_true, bf_size=bf_memory))
    # return 1 - best_fpr


def train_with_fpr(model, train_loader, val_loader, all_memory,
                   all_record, num_epochs=100, output_acc=True):
    model_size = get_model_size(model)
    criterion = fpr_loss(all_record=all_record, all_memory=all_memory, model_size=model_size, epsilon=1e-12)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    val_FNR = (torch.rand(1) * 0.1).to(device)
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

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            # targets = targets.to(device)
            targets = targets.view(-1, 1).to(device)
            targets = torch.tensor(targets, dtype=torch.int32)
            optimizer.zero_grad()
            outputs = model(inputs)
            predicted = get_result(outputs)
            total_samples += targets.size(0)
            targets = torch.tensor(targets, dtype=torch.int64)

            targets_float = targets.float()
            loss = criterion(outputs, targets_float, val_FNR)
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
        # print(f'train_false_positives: {train_false_positives}')
        # print(f'train_false_data_cnt: {train_false_data_cnt}')
        # print(f'train_false_negatives: {train_false_negatives}')
        # print(f'train_true_data_cnt: {train_true_data_cnt}')
        train_FPR = train_false_positives / train_false_data_cnt
        train_FNR = train_false_negatives / train_true_data_cnt

        model.eval()  # 设置模型为评估模式
        val_correct = 0
        val_samples = 0
        val_running_loss = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                # targets = targets.to(device)
                targets = targets.view(-1, 1).to(device)
                targets = torch.tensor(targets)
                outputs = model(inputs)
                predicted = get_result(outputs)
                loss = criterion(outputs, targets.float(), val_FNR)
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


def train_with_fpr_1(model, train_loader, val_loader, bf_memory,
                     true_data_count, num_epochs=100, output_acc=True):
    criterion = fpr_loss_1(true_data_count=true_data_count, bf_memory=bf_memory, epsilon=1e-12)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    val_FNR = (torch.rand(1) * 0.1).to(device)
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

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.view(-1, 1).to(device)
            targets = torch.tensor(targets, dtype=torch.int32)
            optimizer.zero_grad()
            outputs = model(inputs)
            predicted = get_result(outputs)
            total_samples += targets.size(0)
            targets = torch.tensor(targets, dtype=torch.int64)

            targets_float = targets.float()
            loss = criterion(outputs, targets_float, val_FNR)
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

        model.eval()  # 设置模型为评估模式
        val_correct = 0
        val_samples = 0
        val_running_loss = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.view(-1, 1).to(device)
                targets = torch.tensor(targets)
                outputs = model(inputs)
                predicted = get_result(outputs)
                loss = criterion(outputs, targets.float(), val_FNR)
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


def predict_single_row(model, row):
    model.eval()
    with torch.no_grad():
        row = torch.tensor(row, dtype=torch.float32).unsqueeze(0).to(device)
        output = model(row)
        prediction = (output > 0.5).float().cpu().item()
    return prediction


def batch_predict_accuracy(model, X, batch_size=128):
    tensor_data = torch.tensor(X, dtype=torch.float32).to(device)

    dataset = TensorDataset(tensor_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_predictions = []

    model.eval()
    with torch.no_grad():
        for inputs in dataloader:
            inputs = inputs[0]  # unpack the tuple
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            all_predictions.append(predicted.cpu().numpy())

    # Concatenate all predictions into a single numpy array
    all_predictions = np.concatenate(all_predictions).flatten()

    return all_predictions


def validate(model, X_train, y_train, X_test, y_test):
    prediction_results = batch_predict_accuracy(model, X_train)
    indices_train = [i for i in range(len(X_train)) if prediction_results[i] == 0 and y_train[i] == 1]
    prediction_results = batch_predict_accuracy(model, X_test)
    indices_test = [i for i in range(len(X_test)) if prediction_results[i] == 0 and y_test[i] == 1]
    return np.concatenate((X_train[indices_train], X_test[indices_test]), axis=0)


def create_bloom_filter(dataset, bf_name, bf_size):
    n_items = len(dataset)
    print('n_items = ', n_items)
    print('bf_size = ', bf_size)

    # # 计算布隆过滤器的最佳误差率
    # # bf_size 以字节为单位
    # # m 是总的比特数，1 字节 = 8 比特
    # m = bf_size * 8
    # # k = m/n * ln(2)
    # # error_rate = (1 - exp(-k * n/m))^k
    # # 粗略估算最佳 k 值
    # best_k = round((m / n_items) * math.log(2))
    # print('best_k = ', best_k)
    # print(best_k * n_items / m)
    # optimal_error_rate = (1 - math.exp(-best_k * n_items / m)) ** best_k
    # print(f"Optimal error rate: {optimal_error_rate}")

    # # 创建布隆过滤器
    # bloom_filter = BloomFilter(capacity=n_items, error_rate=optimal_error_rate)
    bloom_filter = BloomFilter(capacity=n_items, error_rate=bf_util.get_fpr(n_items, bf_size))
    for data in dataset:
        bloom_filter.add(data)

    with open(bf_name, 'wb') as bf_file:
        pickle.dump(bloom_filter, bf_file)

    print("布隆过滤器已保存")
    return bloom_filter


def query(model, bloom_filter, X_query, y_query):
    fn = 0
    fp = 0

    total = len(X_query)
    print(f"query count = {total}")

    prediction_results = batch_predict_accuracy(model, X_query)
    for i in range(total):
        input_data = X_query[i]
        # true_label = 1 - y_query[i]
        true_label = y_query[i]
        # print(f'true_label = {true_label}')
        # prediction = predict_single_row(model, input_data)
        prediction = prediction_results[i]
        if prediction == 1:
            if true_label == 0:
                fp = fp + 1
        else:
            if input_data in bloom_filter:
                if true_label == 0:
                    fp = fp + 1
            else:
                if true_label == 1:
                    fn = fn + 1
                    # print(i, input_data)

    print(f"fp: {fp}")
    print(f"total: {total}")
    print(f"fpr: {float(fp) / total}")
    print(f"fnr: {float(fn) / total}")
    return float(fp) / total


class Bayes_Optimizer:
    def __init__(self, input_dim, output_dim, train_loader, val_loader, all_record, all_memory, true_data_count,
                 learning_rate=0.001, hidden_units=(8, 512)):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = learning_rate
        self.hidden_units = hidden_units
        self.best_model = None
        self.all_record = all_record
        self.all_memory = all_memory
        self.true_data_count = true_data_count

    def target_function(self, num_hidden_units):
        # 确保 num_hidden_units 是整数
        num_hidden_units = int(num_hidden_units)

        # 初始化模型
        model = SimpleNetwork([num_hidden_units], input_dim=self.input_dim, output_dim=self.output_dim)

        # 训练模型
        return train(model, bf_memory=self.all_memory - get_model_size(model), n_true=self.true_data_count,
                     train_loader=self.train_loader, val_loader=self.val_loader, num_epochs=10, output_acc=False)

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
        print(f"choose {best_num_hidden_units} hidden units")
        return best_model
