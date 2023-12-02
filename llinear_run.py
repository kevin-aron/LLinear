import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def plot_results(y_actual, y_predicted, train_size):
    plt.figure(figsize=(10, 6))
    plt.plot(range(train_size), y_actual[:train_size], label='Train Actual Flow', color='green')
    plt.plot(range(train_size, len(y_actual)), y_actual[train_size:], label='Test Actual Flow', color='blue')
    plt.plot(range(train_size, len(y_actual)), y_predicted, label='Predicted Flow', color='red')
    plt.axvline(x=train_size, linestyle='--', color='gray')
    plt.xlabel('Time')
    plt.ylabel('Flow')
    plt.title('Actual vs Predicted Flow')
    plt.legend()
    plt.show()
# def calculate_mae(y_true, y_pred):
#     y_true = torch.tensor(y_true).clone().detach()
#     y_pred = torch.tensor(y_pred).clone().detach()
#     mae = torch.mean(torch.abs(y_true - y_pred))
#     return mae.item()


file_path = 'dataflow.xlsx'
data = pd.read_excel(file_path)

features = ['year', 'month', 'yq_flow', 'pd_flow', 'mx_flow', 'sy_flow', 'xy_flow', 'hs_flow', 'zq_flow']
target = 'flow'

X = data[features].values  # 特征
y = data[target].values    # 目标值

# 数据转换为PyTorch张量
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# 模型定义
class LLinear(nn.Module):
    def __init__(self, input_size, seq_len, pred_len):
        super(LLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Other = nn.Linear(input_size - 2, self.pred_len)

    def forward(self, x):
        year_month = x[:, :2]  # 提取年份和月份
        other_features = x[:, 2:]  # 其他特征
        seasonal_output = self.Linear_Seasonal(year_month)
        trend_output = self.Linear_Trend(year_month)
        other_output = self.Linear_Other(other_features)
        other_output = other_output.view(-1, self.pred_len)
        x = seasonal_output + trend_output + other_output
        return x

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# 初始化模型并定义损失函数和优化器
input_size = X.shape[1]
seq_len = 2
pred_len = 1

model = LLinear(input_size, seq_len, pred_len)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10000
batch_size = 32

for epoch in range(num_epochs):
    for i in range(0, len(X_train), batch_size):
        optimizer.zero_grad()
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]

        outputs = model(batch_X)
        loss = criterion(outputs.squeeze(), batch_y)
        loss.backward()
        optimizer.step()

    # 在每个 epoch 结束后计算误差值并输出
    with torch.no_grad():
        y_pred = model(X_test).squeeze().numpy()

    mse = mean_squared_error(y_test, y_pred)
    # mae = calculate_mae(y_test, y_pred)
    nse = 1 - (torch.sum(torch.square(y_test - y_pred)) / torch.sum(
        torch.square(y_test - torch.mean(y_test))))  # 计算 NSE
    print(f"Epoch [{epoch+1}/{num_epochs}], MSE: {mse}, NSE: {nse}")

plot_results(y_tensor.numpy(), y_pred, len(X_train))