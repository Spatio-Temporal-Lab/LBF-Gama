import torch
import torch.nn as nn
import torch.optim as optim
import random
import torch.optim as optim
from torch.nn import TransformerEncoderLayer, TransformerEncoder, LayerNorm
from torch.nn.init import xavier_uniform_
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
from memory_profiler import profile
from torchsummary import summary
import pandas as pd
import math
from bayes_opt import BayesianOptimization
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# update逻辑 先通过模型+BF判断对错，然后再通过模型判断数据，插入布隆过滤器


class FPRloss(nn.Module):

    def __init__(self, all_memory, all_record):
        super(FPRloss, self).__init__()
        self.all_memory = all_memory
        self.all_record = all_record

    def forward(self, input, target, network):
        input = get_result(input)
        num_item = len(input)
        FPR = torch.sum((input == 1) & (target == 0)).item() / num_item
        FNR = torch.sum((input == 0) & (target == 1)).item() / num_item
        if FNR == 0:
            FNR = 0.00001
        memory_cost = getModelSize(network)
        memory_cost = (self.all_memory - memory_cost) * 8
        bf_FPR = pow(math.e, -((memory_cost * math.log(2) * math.log(2)) / (self.all_record * FNR)))

        loss = FNR * bf_FPR + FPR
        # loss_tensor = torch.tensor(loss, dtype=torch.float32, requires_grad=True)  # 将损失值转换为张量
        if (loss > 1):
            print('FNR:', FNR)
            print('FPR:', FPR)
            print('bf_FPR:', bf_FPR)
        # print(loss)
        return loss


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


def getModelSize(model):
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
    all_size = all_size_bit / 1024 / 1024
    # print('模型总大小为：{:.3f}MB'.format(all_size))
    return all_size_bit


def train(model, train_loader, val_loader, num_epochs=100, loss_fun=nn.BCELoss(), output_acc=True):
    criterion = loss_fun
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
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
        sum_neg = 0
        cnt = 0
        train_acc_list = []
        train_loss_list = []
        train_FPR_list = []
        train_FNR_list = []
        val_acc_list = []
        val_loss_list = []
        val_FPR_list = []
        val_FNR_list = []

        for inputs, targets in train_loader:
            inputs = inputs.to(device)  ##******#
            targets = targets.to(device)  # ******#
            targets = torch.tensor(targets)
            targets = torch.tensor(targets, dtype=torch.int32)
            optimizer.zero_grad()
            outputs = model(inputs)
            # predicted = torch.round(outputs).int()
            # print("output",outputs)
            predicted = get_result(outputs)
            total_samples += targets.size(0)
            targets = torch.tensor(targets, dtype=torch.int64)
            total_correct += (predicted == targets).sum().item()
            targets_float = targets.float()
            loss = criterion(outputs, targets_float)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
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
                inputs = inputs.to(device)  ##******#
                targets = targets.to(device)  # ******#
                targets = torch.tensor(targets)
                outputs = model(inputs)
                # predicted = torch.round(outputs).int()
                predicted = get_result(outputs)
                loss = criterion(outputs, targets.float())
                val_running_loss += loss.item()
                val_samples += targets.size(0)
                val_correct += (predicted == targets).sum().item()
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
        if output_acc:
            print(
                f"Epoch {epoch + 1} - Loss: {running_loss:.4f} - Train Accuracy: {train_accuracy:.4f} - Val Accuracy: {val_accuracy:.4f}")
            print(
                f"Epoch {epoch + 1} - train_FPR: {train_FPR:.4f} - train_FNR: {train_FNR:.4f} - val_FPR: {val_FPR:.4f} - val_FNR: {val_FNR:.4f}")

    return train_accuracy


def validate(model,word_dict,region_dict):
    pass








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
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        # 训练模型
        return train(model, train_loader=self.train_loader, val_loader=self.val_loader, num_epochs=1,
                     loss_fun=nn.BCELoss(), output_acc=False)

    def optimize(self):
        optimizer = BayesianOptimization(
            f=self.target_function,
            pbounds={'num_hidden_units': self.hidden_units},
            random_state=42,
        )
        optimizer.maximize(n_iter=1)

        best_params = optimizer.max['params']
        best_num_hidden_units = int(best_params['num_hidden_units'])

        # 使用最佳参数重新训练模型
        best_model = SimpleNetwork([best_num_hidden_units], input_dim=self.input_dim, output_dim=self.output_dim)
        print(optimizer.max)
        return best_model
