import pandas as pd

# 读取 CSV 文件
file_path = "/Users/isparkyou/PycharmProjects/home-energy-trading/test/agents/prediction/combined_hourly_data.csv"  # 请替换为你的文件路径
df = pd.read_csv(file_path)

# 确保时间列是 datetime 格式（假设列名是 'date_time'）
df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce')

# 转换为 "YYYYMMDDHH" 格式
df['date_time'] = df['date_time'].dt.strftime('%Y%m%d%H')

# 保存回 CSV 文件
df.to_csv("converted_file.csv", index=False)