import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os

# Define RMSE function
def rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))

# Step 1: Load dataset for model generation
model_data_path = '/Users/isparkyou/PycharmProjects/home-energy-trading/test/agents/prediction/combined_hourly_data_no_time.csv'
if not os.path.exists(model_data_path):
    raise FileNotFoundError(f"Dataset for model generation not found at: {model_data_path}")

model_data = pd.read_csv(model_data_path)

# Select only the required columns: lmd_totalirrad, lmd_temperature, and target column 'power'
model_data = model_data[['lmd_totalirrad', 'lmd_temperature', 'power']]

# Handle missing values
model_data = model_data.dropna()

# Split dataset into features (X) and target (y)
X = model_data[['lmd_totalirrad', 'lmd_temperature']]
y = model_data['power']

# Train-test split for model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train base models
print("Training base models...")

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions_train = rf_model.predict(X_train)

# Support Vector Machine (SVM)
svm_model = SVR(kernel='rbf')
svm_model.fit(X_train, y_train)
svm_predictions_train = svm_model.predict(X_train)

# XGBoost
dtrain = xgb.DMatrix(data=X_train, label=y_train)
xgb_model = xgb.train(params={'objective': 'reg:squarederror', 'eta': 0.1, 'max_depth': 3},
                      dtrain=dtrain, num_boost_round=50)
xgb_predictions_train = xgb_model.predict(dtrain)

# Prepare data for stacking
stacking_train = pd.DataFrame({
    'rf': rf_predictions_train,
    'svm': svm_predictions_train,
    'xgb': xgb_predictions_train,
    'target': y_train
})


# Train meta-model (Linear Regression)
print("Training meta-model for stacking...")
meta_model = LinearRegression()
meta_model.fit(stacking_train[['rf', 'svm', 'xgb']], stacking_train['target'])

# Step 2: Load new data for prediction
prediction_data_path = '/Users/isparkyou/PycharmProjects/home-energy-trading/data/input/solar_hourly_energy_production_dataset.xlsx'
if not os.path.exists(prediction_data_path):
    raise FileNotFoundError(f"Dataset for prediction not found at: {prediction_data_path}")

prediction_data = pd.read_excel(prediction_data_path)

# Select required columns for prediction: lmd_totalirrad and lmd_temperature
prediction_data = prediction_data[['lmd_totalirrad', 'lmd_temperature']]

# Handle missing values in the prediction data
prediction_data = prediction_data.dropna()

# Predict power using the trained stacking model
print("Generating predictions...")
rf_predictions = rf_model.predict(prediction_data)
svm_predictions = svm_model.predict(prediction_data)
xgb_predictions = xgb_model.predict(xgb.DMatrix(data=prediction_data))

stacking_test = pd.DataFrame({
    'rf': rf_predictions,
    'svm': svm_predictions,
    'xgb': xgb_predictions
})

final_predictions = meta_model.predict(stacking_test)

# Step 3: Save predictions to file
output_path = '/Users/isparkyou/PycharmProjects/home-energy-trading/data/output/production/solar_hourly_energy_predictions.xlsx'
prediction_data['predicted_power'] = final_predictions
prediction_data.to_excel(output_path, index=False)

print(f"Predictions saved to: {output_path}")