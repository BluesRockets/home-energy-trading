import pandas as pd
import os
import asyncio
from pathlib import Path
from spade import agent, behaviour
from spade.message import Message
import json


class ProductionSenderAgent(agent.Agent):
    class SendMessageBehaviour(behaviour.CyclicBehaviour):
        def __init__(self, excel_path):
            super().__init__()
            self.excel_path = excel_path
            try:
                self.data = pd.read_excel(self.excel_path)
                self.index = 0
            except Exception as e:
                print(f"Error loading Excel: {e}")
                self.kill()

        async def run(self):
            if self.agent.client is None:
                print("Agent disconnected")
                return

            if self.index < len(self.data):
                row = self.data.iloc[self.index].to_dict()
                try:
                    msg = Message(to="wxu20@xmpp.is")

                    # 重新组织符合图片格式的 JSON
                    message_data = {
                        "type": "production",
                        "timestamp": pd.to_datetime(row["date_time"]).isoformat(),
                        "household_id": row.get("household_id", "default_house"),  # 读取 household_id, 如果无则默认
                        "production": row["predicted_power"]  # 生产的能量值
                    }

                    msg.body = json.dumps(message_data)
                    await self.send(msg)
                    print(f"Sent: {msg.body}")
                    self.index += 1
                except Exception as e:
                    print(f"Send failed: {e}")
            else:
                print("All data sent")
                self.kill()
            await asyncio.sleep(1)

    async def setup(self):
        print("Agent started")
        excel_path = "../../../data/output/production/validation_predictions_with_timestamps.xlsx"
        self.add_behaviour(self.SendMessageBehaviour(excel_path))


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