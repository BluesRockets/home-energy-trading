# model.py
import torch.nn as nn
import torch

class CNN_LSTM_Model(nn.Module):
    def __init__(self):
        super(CNN_LSTM_Model, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=1)
        self.lstm = nn.LSTM(input_size=64, hidden_size=50, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(50, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.conv1(x)
        x, _ = self.lstm(x.permute(0, 2, 1))
        x = torch.relu(self.fc1(x[:, -1, :]))
        x = self.fc2(x)
        return x