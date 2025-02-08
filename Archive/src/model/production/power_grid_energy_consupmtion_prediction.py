import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# **Step 1: 设备选择**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# **Step 2: 读取训练数据**
train_data_path = '/Users/isparkyou/PycharmProjects/home-energy-trading/test/agents/prediction/combined_hourly_data.csv'
if not os.path.exists(train_data_path):
    raise FileNotFoundError(f"Training dataset not found at: {train_data_path}")

# 读取训练数据
train_data = pd.read_csv(train_data_path)
train_data['date_time'] = pd.to_datetime(train_data['date_time'])
train_data = train_data.sort_values(by='date_time')

# 仅保留需要的特征
train_data = train_data[['date_time', 'lmd_totalirrad', 'lmd_temperature', 'power']]
train_data = train_data.dropna()

# **额外特征：添加小时信息**
train_data['hour'] = train_data['date_time'].dt.hour / 23.0  # 归一化到 0-1

# **改成 StandardScaler()**
scaler_X = StandardScaler()
scaler_y = StandardScaler()
train_data[['lmd_totalirrad', 'lmd_temperature']] = scaler_X.fit_transform(train_data[['lmd_totalirrad', 'lmd_temperature']])
train_data[['power']] = scaler_y.fit_transform(train_data[['power']])


# **扩充 totalirrad = 0 的数据**
zero_data = train_data[train_data['lmd_totalirrad'] == 0]
train_data = pd.concat([train_data, zero_data, zero_data])  # 复制两遍，增加权重

# **Step 3: 划分训练集和验证集**
split_index = int(len(train_data) * 0.8)
train_df = train_data.iloc[:split_index]
val_df = train_data.iloc[split_index:]

# **提取特征和目标变量**
X_train = train_df[['lmd_totalirrad', 'lmd_temperature', 'hour']].values
y_train = train_df['power'].values
X_val = val_df[['lmd_totalirrad', 'lmd_temperature', 'hour']].values
y_val = val_df['power'].values

# **转换为 PyTorch Tensors**
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)  # shape (batch_size, 3)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1).to(device)  # shape (batch_size, 1)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1).to(device)


# **Step 4: 定义 CNN-LSTM 模型**
class CNN_LSTM_Model(nn.Module):
    def __init__(self):
        super(CNN_LSTM_Model, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=1, stride=1)
        self.lstm = nn.LSTM(input_size=64, hidden_size=50, num_layers=1, batch_first=True)
        self.layer_norm = nn.LayerNorm(50)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(50, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.layer_norm(x[:, -1, :])
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x  # **取消 Softplus**


# **Step 5: 训练模型**
model = CNN_LSTM_Model().to(device)
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.01)  # **提高初始学习率**
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.0001)

print("Training the model...")
num_epochs = 50
batch_size = 64

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

train_losses, val_losses = [], []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)

        zero_mask = (batch_X[:, 0] == 0).float()
        constraint_loss = torch.mean(zero_mask * outputs ** 2)

        loss = criterion(outputs, batch_y) + 0.1 * constraint_loss  # 仅对 `totalirrad=0` 时加约束
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor).item()

    train_losses.append(epoch_loss / len(train_loader))
    val_losses.append(val_loss)

    print(
        f"Epoch {epoch + 1}/{num_epochs} - Training Loss: {epoch_loss / len(train_loader):.4f} - Validation Loss: {val_loss:.4f}")

    scheduler.step()  # 更新学习率

# **Step 6: 预测新数据**
new_data_path = '/Users/isparkyou/PycharmProjects/home-energy-trading/data/input/solar_hourly_energy_production_dataset_converted.xlsx'
if not os.path.exists(new_data_path):
    raise FileNotFoundError(f"New dataset for prediction not found at: {new_data_path}")

new_data = pd.read_excel(new_data_path)
new_data['date_time'] = pd.to_datetime(new_data['date_time'])

# 处理输入特征
new_data['hour'] = new_data['date_time'].dt.hour / 23.0
X_test = new_data[['lmd_totalirrad', 'lmd_temperature', 'hour']].values
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

# **进行预测**
with torch.no_grad():
    test_predictions = model(X_test_tensor).cpu().numpy().flatten()

# **检查反归一化**
print("Predictions before inverse transform:", test_predictions.max(), test_predictions.min())
# **反归一化 power**
test_predictions = scaler_y.inverse_transform(test_predictions.reshape(-1, 1)).flatten()
# **再次检查反归一化后的值**
print("Predictions after inverse transform:", test_predictions.max(), test_predictions.min())

# **保存预测结果**
output_path = '/Users/isparkyou/PycharmProjects/home-energy-trading/data/output/production/validation_predictions_with_timestamps.xlsx'
new_data['predicted_power'] = test_predictions
new_data = new_data[['date_time', 'lmd_totalirrad', 'lmd_temperature', 'predicted_power']]  # 删除 hour 列
new_data.to_excel(output_path, index=False)

print(f"Predictions saved to: {output_path}")