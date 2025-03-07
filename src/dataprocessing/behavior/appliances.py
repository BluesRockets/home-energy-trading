#XMPP发送
import numpy as np
import pandas as pd
import calendar
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
# 定义获取每月天数的函数
def get_days_in_month(date):
    return calendar.monthrange(date.year, date.month)[1]

# 读取数据集
file_path = '/Users/liuyang/Desktop/Agile project/appliance_consumption.csv'
data = pd.read_csv(file_path)

# 提取时间特征
data['time'] = pd.to_datetime(data['time'])
data['month'] = data['time'].dt.month
data['day'] = data['time'].dt.day
data['hour'] = data['time'].dt.hour
data['is_weekend'] = (data['time'].dt.dayofweek >= 5).astype(int)
data['days_in_month'] = data['time'].apply(get_days_in_month)

# 使用周期性编码处理日期特征
data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
data['day_sin'] = np.sin(2 * np.pi * data['day'] / data['days_in_month'])
data['day_cos'] = np.cos(2 * np.pi * data['day'] / data['days_in_month'])
data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
data.columns = [col.replace(' [kW]', '').strip() for col in data.columns]

# 锁定特征和目标
features = ['Dishwasher', 'Furnace', 'Home office', 'Fridge', 'Wine cellar',
            'Garage door', 'Kitchen', 'Living room', 'temperature', 'month_sin',
            'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'is_weekend']
data_goal = data[features]
target_appliances = features[:8]  # 提取目标家电名称

# 划分训练集和测试集
train_size = int(len(data_goal) * 0.8)
train_data = data_goal[:train_size]
test_data = data_goal[train_size:].copy()

# 手动构造时间列
test_data.loc[:, 'time'] = pd.date_range(start='2024-01-01', periods=len(test_data), freq='h')

X_train = train_data[features].values
X_test = test_data[features].values

y_train = train_data[target_appliances].values
y_test = test_data[target_appliances].values

# 定义 LSTM 输入格式函数
def lstm_input(X, y, timesteps=24):
    X_lstm, y_lstm = [], []
    for i in range(timesteps, len(X)):
        X_lstm.append(X[i - timesteps:i])
        y_lstm.append(y[i])
    return np.array(X_lstm), np.array(y_lstm)

# 构建 LSTM 模型
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

# 训练模型
batch_size = min(16, X_train_lstm.shape[0])
model.fit(X_train_lstm, y_train_lstm, epochs=40, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping])

# 取最后 24 个训练数据
last_24_train_data = X_train[-24:]
# 生成未来 24 小时的输入数据
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
# 确保 future_data 包含所有特征列
for feature in features:
    if feature not in future_data.columns:
        future_data[feature] = 0  # 或者使用其他适当的默认值
# 重新排列列的顺序
future_data = future_data[features]
X_future = future_data.values
# 拼接最后 24 个训练数据和未来数据
X_future_with_history = np.vstack([last_24_train_data, X_future])
# 将输入数据转换为 LSTM 格式
X_future_lstm, _ = lstm_input(X_future_with_history, np.zeros((X_future_with_history.shape[0], y_train.shape[1])), timesteps=24)
# 只取未来 24 小时的预测结果
future_predictions = model.predict(X_future_lstm)[-24:]

# 输出未来 24 小时的预测结果
future_predictions_df = pd.DataFrame(future_predictions, columns=target_appliances)
future_predictions_df[future_predictions_df < 0] = 0
future_predictions_df['time'] = future_dates

import asyncio
import slixmpp
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

        # 启动后台任务来发送消息
        asyncio.create_task(self.send_messages_periodically())

    async def send_messages_periodically(self):
        while True:
            # 拆分长消息为4批发送
            max_message_length = 500  # 每条消息的最大长度
            messages = [self.message[i:i + max_message_length] for i in range(0, len(self.message), max_message_length)]

            # 分批发送
            for msg in messages:
                self.send_message(mto=self.recipient, mbody=msg, mtype='chat')
                await asyncio.sleep(1)  # 每次发送后等待1秒

            # 每24小时发送一次（86400秒）
            await asyncio.sleep(86400)  # 等待24小时后再次发送

    def disconnected(self, event):
        print("Disconnected from XMPP server")

future_predictions_df = future_predictions_df.round(3)
# 将预测数据转换为文本消息
prediction_text = "Household Appliance Power Consumption Prediction:\n"
prediction_text += future_predictions_df.to_string(index=False)

# XMPP 账号信息
xmpp_sender = "appliance@xmpp.is"
xmpp_password = "prediction5014"
xmpp_recipient = "pplively@xmpp.is"  # 替换为目标 XMPP 账号

# 发送 XMPP 消息
xmpp_client = SendPrediction(xmpp_sender, xmpp_password, xmpp_recipient, prediction_text)
xmpp_client.connect()

# 启动客户端事件循环
asyncio.get_event_loop().run_until_complete(xmpp_client.process(forever=True))