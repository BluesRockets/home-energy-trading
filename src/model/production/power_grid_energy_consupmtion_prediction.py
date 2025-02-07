import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score

# **Step 1: 设备选择**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# **Step 2: 读取训练数据**
train_data_path = '/Users/isparkyou/PycharmProjects/home-energy-trading/test/agents/prediction/converted_file.csv'
if not os.path.exists(train_data_path):
    raise FileNotFoundError(f"Training dataset not found at: {train_data_path}")

# 读取训练数据
train_data = pd.read_csv(train_data_path)
train_data['date_time'] = pd.to_datetime(train_data['date_time'])
train_data = train_data.sort_values(by='date_time')
train_data = train_data[['date_time', 'lmd_totalirrad', 'lmd_temperature', 'power']]
train_data = train_data.dropna()

# **Step 3: 划分训练集和验证集**
split_index = int(len(train_data) * 0.8)
train_df = train_data.iloc[:split_index]
val_df = train_data.iloc[split_index:]

# **提取特征和目标变量**
X_train = train_df[['lmd_totalirrad', 'lmd_temperature']].values
y_train = train_df['power'].values
X_val = val_df[['lmd_totalirrad', 'lmd_temperature']].values
y_val = val_df['power'].values

# **转换为 PyTorch Tensors**
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1).to(device)  # (batch_size, 2, 1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1).to(device)

# **Step 4: 定义 CNN-LSTM 模型**
class CNN_LSTM_Model(nn.Module):
    def __init__(self):
        super(CNN_LSTM_Model, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=1)
        self.pool = nn.MaxPool1d(kernel_size=1, stride=1)
        self.lstm = nn.LSTM(input_size=64, hidden_size=50, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(50, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# **Step 5: 训练模型**
model = CNN_LSTM_Model().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training the model...")
num_epochs = 50
batch_size = 32

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor).item()

    print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {epoch_loss/len(train_loader):.4f} - Validation Loss: {val_loss:.4f}")

# **Step 6: 保存训练好的模型**
model_save_path = '/Users/isparkyou/PycharmProjects/home-energy-trading/src/model/production/cnn_lstm_model.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to: {model_save_path}")

# ---------------------------------------------
# **Step 7: 预测新数据**
# ---------------------------------------------
print("Loading model for prediction...")

# **加载模型**
model.load_state_dict(torch.load(model_save_path))
model.eval()

# **读取新数据**
new_data_path = '/Users/isparkyou/PycharmProjects/home-energy-trading/data/input/solar_hourly_energy_production_dataset.xlsx'
if not os.path.exists(new_data_path):
    raise FileNotFoundError(f"New dataset for prediction not found at: {new_data_path}")

# **读取 Excel 并检查列名**
new_data = pd.read_excel(new_data_path, header=0)
print("Columns in new_data:", new_data.columns.tolist())  # ✅ 先检查列名

# **修正列名**
new_data.rename(columns=lambda x: x.strip().lower(), inplace=True)  # ✅ 去除空格并小写
print("Renamed columns:", new_data.columns.tolist())

# **检查是否有 `date_time` 列**
if 'date_time' not in new_data.columns:
    raise KeyError("Error: Column 'date_time' not found in dataset! Check file format.")

# **转换时间格式**
new_data['date_time'] = pd.to_datetime(new_data['date_time'])

# **检查数据是否正确加载**
print("Shape of new_data:", new_data.shape)  # ✅ 预期 shape=(N, 3)
new_data['date_time'] = pd.to_datetime(new_data['date_time'])
new_data = new_data.sort_values(by='date_time')
new_data = new_data[['date_time', 'lmd_totalirrad', 'lmd_temperature']]
new_data = new_data.dropna()

# **转换为 PyTorch Tensor**
X_test = new_data[['lmd_totalirrad', 'lmd_temperature']].values
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1).to(device)

# **进行预测**
with torch.no_grad():
    test_predictions = model(X_test_tensor).cpu().numpy().flatten()

# **保存预测结果**
output_path = '/Users/isparkyou/PycharmProjects/home-energy-trading/data/output/production/validation_predictions_with_timestamps.xlsx'
new_data['predicted_power'] = test_predictions
new_data[['date_time', 'predicted_power']].to_excel(output_path, index=False)

print(f"Predictions saved to: {output_path}")