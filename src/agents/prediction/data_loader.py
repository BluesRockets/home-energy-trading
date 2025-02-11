# data_loader.py
import pandas as pd
import torch


def load_data(file_path):
    """ 加载数据并返回特征和目标变量 """
    df = pd.read_csv(file_path)
    df['date_time'] = pd.to_datetime(df['date_time'])
    df = df.sort_values(by='date_time')

    # 选取特征
    X = df[['lmd_totalirrad', 'lmd_temperature']].values
    y = df['power'].values

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)