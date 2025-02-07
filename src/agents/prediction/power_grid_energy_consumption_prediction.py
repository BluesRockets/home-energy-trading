import pandas as pd
import numpy as np
import os

from keras.src.layers import LSTM, Conv1D, MaxPooling1D, Dropout, Dense
from tensorflow.python.keras import Sequential
# from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, LSTM, Dropout, Dense
# from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping
#from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, Dropout, Dense
from tensorflow.python.keras.optimizer_v1 import Adam


# Define RMSE function (without scikit-learn)
def rmse(actual, predicted):
    error = np.subtract(actual, predicted)
    sq_error = np.square(error)
    mean_sq_error = np.mean(sq_error)
    rmse_value = np.sqrt(mean_sq_error)
    return rmse_value


# Define R-squared function (without scikit-learn)
def r2_score_fun(y_true, y_pred):
    ss_total = np.sum(np.square(y_true - np.mean(y_true)))
    ss_residual = np.sum(np.square(y_true - y_pred))
    r2_value = 1 - (ss_residual / ss_total)
    return r2_value


# Step 1: Load dataset for model generation
data_path = '/Users/isparkyou/PycharmProjects/home-energy-trading/test/agents/prediction/combined_hourly_data.csv'
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset for model generation not found at: {data_path}")

# Load data
model_data = pd.read_csv(data_path)

# Step 2: Preprocess the dataset
# Ensure the 'date_time' column is in datetime format
model_data['date_time'] = pd.to_datetime(model_data['date_time'])

# Sort by date_time (just in case data is not sorted)
model_data = model_data.sort_values(by='date_time')

# Select required columns: lmd_totalirrad, lmd_temperature, and power
model_data = model_data[['date_time', 'lmd_totalirrad', 'lmd_temperature', 'power']]

# Handle missing values
model_data = model_data.dropna()

# Step 3: Split data into training and validation sets (80% train, 20% validation)
split_index = int(len(model_data) * 0.8)
train_data = model_data.iloc[:split_index]
validation_data = model_data.iloc[split_index:]

# Extract features and target from training and validation sets
X_train = train_data[['lmd_totalirrad', 'lmd_temperature']].values
y_train = train_data['power'].values

X_val = validation_data[['lmd_totalirrad', 'lmd_temperature']].values
y_val = validation_data['power'].values

# Reshape data for CNN and LSTM (3D input for time-series models)
X_train_cnn = np.expand_dims(X_train, axis=1)  # Add 1 timestep
X_val_cnn = np.expand_dims(X_val, axis=1)

# Step 4: Define the CNN-LSTM model
print("Building CNN-LSTM model...")
model = Sequential([
    Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(1, X_train.shape[1])),
    MaxPooling1D(pool_size=2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1)  # Output layer for regression
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Step 5: Train the model
print("Training the model...")
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train_cnn, y_train,
    validation_data=(X_val_cnn, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate model performance on validation set
val_predictions = model.predict(X_val_cnn).flatten()

# Step 6: Calculate Accuracy Metrics
val_mse = np.mean((y_val - val_predictions) ** 2)
val_rmse = rmse(y_val, val_predictions)
val_r2 = r2_score_fun(y_val, val_predictions)

print(f"Validation MSE: {val_mse:.4f}")
print(f"Validation RMSE: {val_rmse:.4f}")
print(f"Validation R² (Accuracy): {val_r2:.4f}")

# Step 7: Save predictions with timestamps for validation set
output_path = '/data/output/production/validation_predictions_with_timestamps.xlsx'

# Combine predictions with timestamps
validation_data['predicted_power'] = val_predictions
validation_data[['date_time', 'power', 'predicted_power']].to_excel(output_path, index=False)

print(f"Validation predictions saved to: {output_path}")
