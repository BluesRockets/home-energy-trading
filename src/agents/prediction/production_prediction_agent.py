import pandas as pd
import os
import asyncio
from spade import agent, behaviour
from spade.message import Message
import json


class ProductionSenderAgent(agent.Agent):
    class SendMessageBehaviour(behaviour.CyclicBehaviour):
        def __init__(self, excel_paths):
            super().__init__()
            self.excel_paths = excel_paths
            self.household_data = {}  # 存储所有 household 的数据
            self.household_indices = {}  # 记录每个 household 的发送进度
            self.load_data()  # 读取数据

        def load_data(self):
            """ 读取 4 个 Excel 文件，每个文件有 25 个 Sheets，并将 household_id 转换为纯数字 """
            try:
                for file_path in self.excel_paths:
                    excel_data = pd.ExcelFile(file_path)
                    for sheet in excel_data.sheet_names:
                        try:
                            # 提取数字部分，假设所有 sheet_name 格式是 house_XXX
                            household_id = int(sheet.replace("house_", ""))  # 去掉 'house_' 并转换为整数
                        except ValueError:
                            print(f"Skipping sheet {sheet} in {file_path} (not a valid numeric ID).")
                            continue

                        df = pd.read_excel(excel_data, sheet_name=sheet)
                        if "date_time" in df.columns and "predicted_power" in df.columns:
                            self.household_data[household_id] = df  # 以整数作为键
                            self.household_indices[household_id] = 0  # 初始化索引
                        else:
                            print(f"Skipping sheet {sheet} in {file_path} due to missing columns.")
            except Exception as e:
                print(f"Error loading Excel files: {e}")
                self.kill()

        async def run(self):
            if self.agent.client is None:
                print("Agent disconnected")
                return

            send_count = 0  # 记录本轮发送的消息数

            for household_id, df in self.household_data.items():
                index = self.household_indices[household_id]

                if index < len(df):
                    row = df.iloc[index].to_dict()
                    try:
<<<<<<< Updated upstream
                        msg = Message(to="loganyang@xmpp.is")  # 目标 XMPP 地址
=======
                        msg = Message(to="wxu20@xmpp.is")  # 目标 XMPP 地址 # "wxu20@xmpp.is"
>>>>>>> Stashed changes

                        # 组织 JSON 消息
                        message_data = {
                            "type": "production",
                            "timestamp": pd.to_datetime(row["date_time"]).isoformat(),
                            "household_id": household_id,  # 现在是整数
                            "production": row["predicted_power"]
                        }

                        msg.body = json.dumps(message_data)
                        await self.send(msg)
                        print(f"Sent: {msg.body}")
                        self.household_indices[household_id] += 1  # 更新索引
                        send_count += 1

                    except Exception as e:
                        print(f"Send failed: {e}")

            if send_count == 0:
                print("All data sent")
                self.kill()

            await asyncio.sleep(1)  # 每秒执行一次

    async def setup(self):
        print("Agent started")
        excel_paths = [
            "../../../data/output/production/validation_predictions_part_1.xlsx",
            "../../../data/output/production/validation_predictions_part_2.xlsx",
            "../../../data/output/production/validation_predictions_part_3.xlsx",
            "../../../data/output/production/validation_predictions_part_4.xlsx"
        ]
        self.add_behaviour(self.SendMessageBehaviour(excel_paths))


async def main():
    sender = ProductionSenderAgent(
        "loganyang@xmpp.is",
        "loganyang123"
    )
    await sender.start()

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await sender.stop()
        print("Agent stopped")


if __name__ == "__main__":
    asyncio.run(main())