import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# **Step 1: 读取训练数据**
train_data_path = "/Users/isparkyou/PycharmProjects/home-energy-trading/test/agents/prediction/combined_hourly_data.csv"
if not os.path.exists(train_data_path):
    raise FileNotFoundError(f"Training dataset not found at: {train_data_path}")

train_data = pd.read_csv(train_data_path)
train_data["date_time"] = pd.to_datetime(train_data["date_time"])

# **Step 2: 选择特征与目标**
features = ["lmd_totalirrad", "lmd_temperature"]
target = "power"

X = train_data[features]
y = train_data[target]

# **Step 3: 处理 `NaN`**
X = X.fillna(X.mean())
y = y.fillna(y.mean())

# **Step 4: 增加 `totalirrad=0` 的数据**
zero_data = train_data[train_data["lmd_totalirrad"] == 0].copy()
zero_data["power"] = 0  # 设定 power=0，强化模型学习
train_data = pd.concat([train_data, zero_data, zero_data, zero_data])  # 复制 3 次，提高权重

# **Step 5: 训练集 & 验证集**
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# **Step 6: 特征工程**
X_train["lmd_totalirrad_sq"] = X_train["lmd_totalirrad"] ** 2  # 添加平方项
X_train["interaction"] = X_train["lmd_totalirrad"] * X_train["lmd_temperature"]  # 交互项
X_val["lmd_totalirrad_sq"] = X_val["lmd_totalirrad"] ** 2
X_val["interaction"] = X_val["lmd_totalirrad"] * X_val["lmd_temperature"]

# **Step 7: 标准化**
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# **Step 8: 定义 RMSE 评估函数**
def rmse(y_actual, y_pred):
    return np.sqrt(mean_squared_error(y_actual, y_pred))

# **Step 9: 训练模型**
# 1️⃣ **支持向量机（SVM）**
svm_model = SVR()
svm_model.fit(X_train_scaled, y_train)
svm_preds = svm_model.predict(X_val_scaled)

# 2️⃣ **随机森林（Random Forest）**
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_preds = rf_model.predict(X_val_scaled)

# 3️⃣ **XGBoost**
dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
dval = xgb.DMatrix(X_val_scaled)
xgb_model = xgb.train({"objective": "reg:squarederror", "eta": 0.1, "max_depth": 3}, dtrain, num_boost_round=50)
xgb_preds = xgb_model.predict(dval)

# **Step 10: 加权平均预测**
weights = {"XGBoost": 0.4, "SVM": 0.3, "Random Forest": 0.3}
combined_preds = (weights["XGBoost"] * xgb_preds) + (weights["SVM"] * svm_preds) + (weights["Random Forest"] * rf_preds)

# **Step 11: 评估模型**
val_rmse = rmse(y_val, combined_preds)
val_mae = mean_absolute_error(y_val, combined_preds)
val_r2 = r2_score(y_val, combined_preds)

print(f"✅ Validation RMSE: {val_rmse:.4f}")
print(f"✅ Validation MAE: {val_mae:.4f}")
print(f"✅ Validation R²: {val_r2:.4f}")

# **Step 12: 训练结果可视化**
plt.figure(figsize=(12, 5))
plt.scatter(y_val, combined_preds, alpha=0.5, label="Predicted vs. Actual")
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r', lw=2, label="Ideal Fit")
plt.xlabel("Actual Power Output")
plt.ylabel("Predicted Power Output")
plt.title("Validation: Actual vs Predicted Power")
plt.legend()
plt.show()

# **Step 13: `totalirrad=0` 预测效果**
zero_mask = X_val["lmd_totalirrad"] == 0
zero_actual = y_val[zero_mask]
zero_preds = combined_preds[zero_mask]

plt.figure(figsize=(10, 5))
plt.scatter(zero_actual, zero_preds, alpha=0.5, label="TotalIrrad = 0 Predictions")
plt.axhline(0, color='red', linestyle="--", label="Ideal: Power = 0")
plt.xlabel("Actual Power Output (TotalIrrad=0)")
plt.ylabel("Predicted Power Output (TotalIrrad=0)")
plt.title("Validation: TotalIrrad=0 Effect")
plt.legend()
plt.show()

# **Step 14: 预测新数据**
input_data_path = "/Users/isparkyou/PycharmProjects/home-energy-trading/data/input/solar_hourly_energy_production_dataset_converted.xlsx"
if not os.path.exists(input_data_path):
    raise FileNotFoundError(f"New dataset for prediction not found at: {input_data_path}")

new_data = pd.read_excel(input_data_path)
new_data["date_time"] = pd.to_datetime(new_data["date_time"])

# 处理输入特征
X_test = new_data[features]
X_test = X_test.fillna(X_test.mean())
X_test["lmd_totalirrad_sq"] = X_test["lmd_totalirrad"] ** 2
X_test["interaction"] = X_test["lmd_totalirrad"] * X_test["lmd_temperature"]
X_test_scaled = scaler.transform(X_test)

# **进行预测**
dtest = xgb.DMatrix(X_test_scaled)
svm_test_preds = svm_model.predict(X_test_scaled)
rf_test_preds = rf_model.predict(X_test_scaled)
xgb_test_preds = xgb_model.predict(dtest)

# **加权平均预测**
final_predictions = (weights["XGBoost"] * xgb_test_preds) + (weights["SVM"] * svm_test_preds) + (weights["Random Forest"] * rf_test_preds)

# **Step 15: 保存预测结果**
output_path = "/Users/isparkyou/PycharmProjects/home-energy-trading/data/output/production/validation_predictions_with_timestamps.xlsx"
new_data["predicted_power"] = final_predictions
new_data.to_excel(output_path, index=False)

print(f"✅ Predictions saved successfully to: {output_path}")