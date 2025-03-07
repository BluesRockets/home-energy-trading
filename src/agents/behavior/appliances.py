#XMPP
import numpy as np
import pandas as pd
import calendar
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
# Define a function to get the number of days per month
def get_days_in_month(date):
    return calendar.monthrange(date.year, date.month)[1]

# Read the dataset
file_path = 'data/input/appliance_consumption.csv'
data = pd.read_csv(file_path)

# Extract time features
data['time'] = pd.to_datetime(data['time'])
data['month'] = data['time'].dt.month
data['day'] = data['time'].dt.day
data['hour'] = data['time'].dt.hour
data['is_weekend'] = (data['time'].dt.dayofweek >= 5).astype(int)
data['days_in_month'] = data['time'].apply(get_days_in_month)

# Use periodic encoding to handle date features
data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
data['day_sin'] = np.sin(2 * np.pi * data['day'] / data['days_in_month'])
data['day_cos'] = np.cos(2 * np.pi * data['day'] / data['days_in_month'])
data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
data.columns = [col.replace(' [kW]', '').strip() for col in data.columns]

# Locking features and targets
features = ['Dishwasher', 'Furnace', 'Home office', 'Fridge', 'Wine cellar',
            'Garage door', 'Kitchen', 'Living room', 'temperature', 'month_sin',
            'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'is_weekend']
data_goal = data[features]
target_appliances = features[:8]

# Divide into training set and test set
train_size = int(len(data_goal) * 0.8)
train_data = data_goal[:train_size]
test_data = data_goal[train_size:].copy()

# Manually construct a time column
test_data.loc[:, 'time'] = pd.date_range(start='2024-01-01', periods=len(test_data), freq='h')

X_train = train_data[features].values
X_test = test_data[features].values

y_train = train_data[target_appliances].values
y_test = test_data[target_appliances].values

# Define LSTM input format function
def lstm_input(X, y, timesteps=24):
    X_lstm, y_lstm = [], []
    for i in range(timesteps, len(X)):
        X_lstm.append(X[i - timesteps:i])
        y_lstm.append(y[i])
    return np.array(X_lstm), np.array(y_lstm)

# LSTM
X_train_lstm, y_train_lstm = lstm_input(X_train, y_train, timesteps=24)
X_test_lstm, y_test_lstm = lstm_input(X_test, y_test, timesteps=24)

model = Sequential()
model.add(Bidirectional(LSTM(units=64, return_sequences=True), input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(units=32, return_sequences=False)))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(y_train_lstm.shape[1]))
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='mean_squared_error')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.summary()

# train the model
batch_size = min(16, X_train_lstm.shape[0])
model.fit(X_train_lstm, y_train_lstm, epochs=40, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping])

# Get the last 24 training data
last_24_train_data = X_train[-24:]
# Generate input data for the next 24 hours
future_dates = pd.date_range(start=test_data['time'].max(), periods=24, freq='h')
future_data = pd.DataFrame({
    'time': future_dates,
    'hour': future_dates.hour,
    'day': future_dates.day,
    'day_of_week': future_dates.dayofweek,
    'month': future_dates.month
})
future_data['is_weekend'] = (future_data['day_of_week'] >= 5).astype(int)
future_data['days_in_month'] = future_data['time'].apply(get_days_in_month)
future_data['month_sin'] = np.sin(2 * np.pi * future_data['month'] / 12)
future_data['month_cos'] = np.cos(2 * np.pi * future_data['month'] / 12)
future_data['day_sin'] = np.sin(2 * np.pi * future_data['day'] / future_data['days_in_month'])
future_data['day_cos'] = np.cos(2 * np.pi * future_data['day'] / future_data['days_in_month'])
future_data['hour_sin'] = np.sin(2 * np.pi * future_data['hour'] / 24)
future_data['hour_cos'] = np.cos(2 * np.pi * future_data['hour'] / 24)
# Ensure that future_data contains all feature columns
for feature in features:
    if feature not in future_data.columns:
        future_data[feature] = 0
# Rearrange the order of columns
future_data = future_data[features]
X_future = future_data.values
# Concatenate the last 24 training data and future data
X_future_with_history = np.vstack([last_24_train_data, X_future])
# Convert input data to LSTM format
X_future_lstm, _ = lstm_input(X_future_with_history, np.zeros((X_future_with_history.shape[0], y_train.shape[1])), timesteps=24)
# Only take the forecast results for the next 24 hours
future_predictions = model.predict(X_future_lstm)[-24:]

# Output the forecast results for the next 24 hours
future_predictions_df = pd.DataFrame(future_predictions, columns=target_appliances)
future_predictions_df[future_predictions_df < 0] = 0
future_predictions_df['time'] = future_dates

import asyncio
import slixmpp
import json

# Convert predictions to structured JSON format
def format_predictions_to_json(future_predictions_df):
    predictions_list = []
    for index, row in future_predictions_df.iterrows():
        for appliance in target_appliances:
            prediction = {
                "timestamp": future_predictions_df['time'].iloc[index].strftime('%Y-%m-%dT%H:%M:%SZ'),
                "type": appliance,
                "production": round(row[appliance], 3),
                "kwh": round(row[appliance], 3)  # Assuming 'production' is in kWh
            }
            predictions_list.append(prediction)
    return json.dumps(predictions_list, indent=4)

# XMPP message sending class
class SendPrediction(slixmpp.ClientXMPP):
    def __init__(self, jid, password, recipient, message):
        super().__init__(jid, password)
        self.recipient = recipient
        self.message = message
        self.add_event_handler("session_start", self.start)
        self.add_event_handler("disconnected", self.disconnected)

    async def start(self, event):
        self.send_presence()
        await self.get_roster()
        # Start a background task to send messages
        asyncio.create_task(self.send_messages_periodically())

    async def send_messages_periodically(self):
        while True:
            # Split long messages into 4 batches and send them
            max_message_length = 500  # the longest message length
            messages = [self.message[i:i + max_message_length] for i in range(0, len(self.message), max_message_length)]

            # Send in batches
            for msg in messages:
                self.send_message(mto=self.recipient, mbody=msg, mtype='chat')
                await asyncio.sleep(1)  # Wait 1 second after each send

            # Send once every 24 hours (86400 seconds)
            await asyncio.sleep(86400)  # Wait 24 hours and send again

    def disconnected(self, event):
        print("Disconnected from XMPP server")

# Format prediction data into JSON
future_predictions_df = future_predictions_df.round(3)
prediction_json = format_predictions_to_json(future_predictions_df)

# XMPP Account Information
xmpp_sender = "appliance@xmpp.is"
xmpp_password = "prediction5014"
xmpp_recipient = "wxu20@xmpp.is"  # Replace with the target XMPP account

# Sending XMPP Messages
xmpp_client = SendPrediction(xmpp_sender, xmpp_password, xmpp_recipient, prediction_json)
xmpp_client.connect()

# Start the client event loop
asyncio.get_event_loop().run_until_complete(xmpp_client.process(forever=True))
