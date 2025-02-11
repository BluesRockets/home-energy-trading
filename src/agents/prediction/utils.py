# utils.py
import os
import logging
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


# 日志配置
def setup_logging(log_file="logs/agent.log"):
    """ 配置日志记录 """
    if not os.path.exists("logs"):
        os.makedirs("logs")

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="a"
    )
    return logging.getLogger(__name__)


# 归一化（StandardScaler 版本）
def normalize_data(df, columns):
    """ 使用 StandardScaler 进行标准化 """
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df, scaler


# 反归一化
def inverse_transform(scaler, predictions):
    """ 反归一化预测结果 """
    return scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()


# 计算模型评估指标
def evaluate_model(y_true, y_pred):
    """ 计算 RMSE 和 R² 分数 """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"MSE": mse, "RMSE": rmse, "R²": r2}


# 绘制训练损失曲线
def plot_loss(train_losses, val_losses):
    """ 绘制损失曲线 """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss", color='blue')
    plt.plot(val_losses, label="Validation Loss", color='red')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.grid()
    plt.show()


# 保存 PyTorch 模型
def save_model(model, model_path):
    """ 保存 PyTorch 模型 """
    torch.save(model.state_dict(), model_path)
    print(f"✅ 模型已保存至: {model_path}")


# 加载 PyTorch 模型
def load_model(model, model_path, device):
    """ 加载 PyTorch 模型 """
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"✅ 模型已加载: {model_path}")
    return model