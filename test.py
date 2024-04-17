import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from bayes_opt import BayesianOptimization

# 加载数据集
iris = load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 将数据转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)


# 定义神经网络模型
class Model(nn.Module):
    def __init__(self, input_size, num_hidden_units):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, num_hidden_units)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(num_hidden_units, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 定义目标函数
def target_function(learning_rate, num_hidden_units):
    model = Model(X.shape[1], int(num_hidden_units))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    for epoch in range(10):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    # 在测试集上评估模型
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test_tensor).float().mean().item()

    return accuracy


# 创建贝叶斯优化对象，并指定参数范围
optimizer = BayesianOptimization(
    f=target_function,
    pbounds={'learning_rate': (0.001, 0.01),
             'num_hidden_units': (10, 50)},
    random_state=42,
)

# 开始优化，指定迭代次数
optimizer.maximize(init_points=5, n_iter=10)

# 打印最优参数和最大值
print(optimizer.max)
