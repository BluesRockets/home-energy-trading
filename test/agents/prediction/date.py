import pandas as pd

# 读取上传的 Excel 文件
file_path = "/Users/isparkyou/PycharmProjects/home-energy-trading/data/input/solar_hourly_energy_production_dataset.xlsx"
df = pd.read_excel(file_path)

# 去除列名中的空格，防止列名不匹配
df.columns = df.columns.str.strip()

# 确保第一列为 'date'，第二列为 'time'
df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0].astype(str))  # 转换日期格式
df.iloc[:, 1] = df.iloc[:, 1].astype(str).str.zfill(2)  # 确保时间格式为 2 位（例如 00, 01, ..., 23）

# 合并 'date' 和 'time' 列
df["date_time"] = df.iloc[:, 0].astype(str) + " " + df.iloc[:, 1] + ":00:00"

# 删除原来的 'date' 和 'time' 列
df = df.drop(df.columns[:2], axis=1)

# 重新排列列顺序
df = df[["date_time"] + list(df.columns[:-1])]

# 保存修改后的文件
output_path = "/Users/isparkyou/PycharmProjects/home-energy-trading/data/input/solar_hourly_energy_production_dataset_converted.xlsx"
df.to_excel(output_path, index=False)
