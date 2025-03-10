import asyncio
import pandas as pd
import os
import time
from spade import agent, behaviour
from spade.message import Message
import json
from datetime import datetime, timedelta

# 设置发送模式参数
SEND_INTERVAL = 1  # 每批数据的发送周期（秒）
HOURS_STEP = 1  # 每次向前移动1小时
MAX_CONCURRENT_TASKS = 20  # 最大并发任务数，避免创建过多任务导致资源耗尽


class HouseholdConsumptionAgent(agent.Agent):
    class SendDataBehaviour(behaviour.CyclicBehaviour):
        def __init__(self):
            super().__init__()
            # 加载数据文件
            self.data = self.load_household_data()
            # 追踪当前处理的小时索引
            self.current_hour_index = 0

            # 添加缓存字典，用于存储已计算的未来预测数据
            self.future_data_cache = {}

            if self.data is not None:
                self.unique_hours = sorted(self.data['time'].unique())
                # 将household_id转换为标准Python整数
                self.data['household_id'] = self.data['household_id'].apply(lambda x: int(str(x).split('_')[-1]))
                self.household_ids = sorted(self.data['household_id'].unique())
            else:
                self.unique_hours = []
                self.household_ids = []

        def load_household_data(self):
            try:
                # Read CSV files directly - now loading all three files
                # 使用GitHub版本的路径格式
                script_dir = os.path.dirname(__file__)
                df1 = pd.read_csv(os.path.abspath(os.path.join(script_dir, '../../../data/output/appliances/household_data1.csv')))
                df2 = pd.read_csv(os.path.abspath(os.path.join(script_dir, '../../../data/output/appliances/household_data2.csv')))
                df3 = pd.read_csv(os.path.abspath(os.path.join(script_dir, '../../../data/output/appliances/household_data3.csv')))

                # 确保时间列是datetime格式
                df1['time'] = pd.to_datetime(df1['time'])
                df2['time'] = pd.to_datetime(df2['time'])
                df3['time'] = pd.to_datetime(df3['time'])

                # 合并数据
                combined_df = pd.concat([df1, df2, df3], ignore_index=True)
                return combined_df

            except FileNotFoundError as e:
                return None
            except Exception as e:
                return None

        def get_future_hours_data(self, timestamp, household_id, appliance, hours=23):
            """从CSV数据中获取未来指定小时数的预测数据"""
            # 生成当前时间点的缓存键
            current_key = f"{timestamp}_{household_id}_{appliance}"

            # 检查是否已缓存此数据
            if current_key in self.future_data_cache:
                return self.future_data_cache[current_key]

            # 计算未来数据
            future_data = []
            future_times = [timestamp + timedelta(hours=i + 1) for i in range(hours)]

            for future_time in future_times:
                future_row = self.data[(self.data['time'] == future_time) &
                                       (self.data['household_id'] == household_id)]

                if not future_row.empty and appliance in future_row.columns:
                    value = round(float(future_row[appliance].iloc[0]), 3)
                    future_data.append(value)
                else:
                    future_data.append(0.0)

            # 更新缓存
            self.future_data_cache[current_key] = future_data

            return future_data

        def format_household_message(self, timestamp, household_id, row):
            """将家庭数据格式化为JSON消息，包含未来预测"""
            # 获取设备列
            appliance_columns = [col for col in self.data.columns if col not in ['time', 'household_id']]

            # 创建JSON结构
            message_dict = {
                "type": "behavior",
                "timestamp": timestamp.strftime('%Y-%m-%dT%H:%M:%S') + 'Z',
                "household_id": int(household_id),  # 确保是标准Python整数
                "data": {}
            }

            # 添加当前和未来的设备读数
            for appliance in appliance_columns:
                if appliance in row and not pd.isna(row[appliance]):
                    # 确保转换为标准Python数据类型
                    current_value = round(float(row[appliance]), 3)
                    # 从缓存或CSV数据中获取未来23小时的预测值
                    future_values = self.get_future_hours_data(timestamp, household_id, appliance)
                    # 当前值加上未来23小时的预测值
                    message_dict["data"][appliance] = [current_value] + future_values

            # 转换为JSON字符串
            return json.dumps(message_dict)

        async def process_household(self, household_id, current_hour):
            """处理单个家庭的数据（可并行执行）"""
            try:
                # 筛选当前小时和家庭的数据
                hour_data = self.data[(self.data['time'] == current_hour) &
                                      (self.data['household_id'] == household_id)]

                if not hour_data.empty:
                    # 格式化包含该家庭所有设备的消息
                    message_text = self.format_household_message(
                        current_hour, household_id, hour_data.iloc[0])

                    # 使用GitHub版本的接收者地址
                    msg = Message(to="wxu20@xmpp.is")
                    msg.body = message_text
                    await self.send(msg)

                    # 直接打印消息
                    print(message_text)

                    return True
                return False
            except Exception as e:
                return False

        async def run(self):
            # 如果没有数据，则终止
            if self.data is None or not self.unique_hours:
                self.kill()
                return

            # 获取当前要处理的小时
            current_idx = self.current_hour_index % len(self.unique_hours)
            current_hour = self.unique_hours[current_idx]

            # 并行处理所有家庭数据
            household_chunks = [self.household_ids[i:i + MAX_CONCURRENT_TASKS]
                                for i in range(0, len(self.household_ids), MAX_CONCURRENT_TASKS)]

            for chunk in household_chunks:
                chunk_tasks = []
                for household_id in chunk:
                    task = asyncio.create_task(self.process_household(household_id, current_hour))
                    chunk_tasks.append(task)

                # 等待当前批次的任务完成
                await asyncio.gather(*chunk_tasks)

            # 移动到下一个小时
            self.current_hour_index = (self.current_hour_index + 1) % len(self.unique_hours)

            # 等待发送间隔
            await asyncio.sleep(SEND_INTERVAL)

    async def setup(self):
        send_behavior = self.SendDataBehaviour()
        self.add_behaviour(send_behavior)


async def main():
    # 创建带有凭据的代理
    sender = HouseholdConsumptionAgent("appliance@xmpp.is", "prediction5014")

    # 启动代理
    await sender.start()

    try:
        # 保持代理运行，直到用户中断
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("程序已停止")
    finally:
        # 停止代理
        await sender.stop()


if __name__ == "__main__":
    asyncio.run(main())