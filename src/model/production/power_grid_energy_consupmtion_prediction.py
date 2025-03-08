import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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

# **Step 4: 训练集 & 验证集**
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# **Step 5: 特征工程**
X_train["lmd_totalirrad_sq"] = X_train["lmd_totalirrad"] ** 2
X_train["interaction"] = X_train["lmd_totalirrad"] * X_train["lmd_temperature"]
X_val["lmd_totalirrad_sq"] = X_val["lmd_totalirrad"] ** 2
X_val["interaction"] = X_val["lmd_totalirrad"] * X_val["lmd_temperature"]

# **Step 6: 标准化**
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# **Step 7: 训练模型**
svm_model = SVR()
svm_model.fit(X_train_scaled, y_train)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
dval = xgb.DMatrix(X_val_scaled)
xgb_model = xgb.train({"objective": "reg:squarederror", "eta": 0.1, "max_depth": 3}, dtrain, num_boost_round=50)

# **Step 8: 读取新数据**
input_data_path = "/Users/isparkyou/PycharmProjects/home-energy-trading/data/input/solar_hourly_energy_production_dataset_converted.xlsx"
if not os.path.exists(input_data_path):
    raise FileNotFoundError(f"New dataset for prediction not found at: {input_data_path}")

new_data = pd.read_excel(input_data_path)
new_data["date_time"] = pd.to_datetime(new_data["date_time"])

# **Step 9: 生成 100 户数据**
household_ids = [f"house_{i:03d}" for i in range(1, 101)]
all_households_data = pd.concat([new_data.assign(household_id=household) for household in household_ids])

# **Step 10: 处理特征**
X_test = all_households_data[features]
X_test = X_test.fillna(X_test.mean())
X_test["lmd_totalirrad_sq"] = X_test["lmd_totalirrad"] ** 2
X_test["interaction"] = X_test["lmd_totalirrad"] * X_test["lmd_temperature"]
X_test_scaled = scaler.transform(X_test)

# **Step 11: 预测**
dtest = xgb.DMatrix(X_test_scaled)
svm_test_preds = svm_model.predict(X_test_scaled)
rf_test_preds = rf_model.predict(X_test_scaled)
xgb_test_preds = xgb_model.predict(dtest)

# **Step 12: 加权平均预测**
weights = {"XGBoost": 0.4, "SVM": 0.3, "Random Forest": 0.3}
all_households_data["predicted_power"] = (
    weights["XGBoost"] * xgb_test_preds +
    weights["SVM"] * svm_test_preds +
    weights["Random Forest"] * rf_test_preds
)

# **Step 13: 分割 100 户家庭成 4 组，每组 25 户**
household_groups = np.array_split(household_ids, 4)

# **Step 14: 保存 4 个 Excel 文件，每个 25 个 Sheet**
output_dir = "/Users/isparkyou/PycharmProjects/home-energy-trading/data/output/production/"
os.makedirs(output_dir, exist_ok=True)

for i, household_group in enumerate(household_groups, start=1):
    output_path = os.path.join(output_dir, f"validation_predictions_part_{i}.xlsx")

    # 以 `ExcelWriter` 方式保存多个 Sheet
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for household_id in household_group:
            household_data = all_households_data[all_households_data["household_id"] == household_id]
            household_data.to_excel(writer, sheet_name=household_id, index=False)

    print(f"✅ Part {i} saved with 25 sheets: {output_path}")