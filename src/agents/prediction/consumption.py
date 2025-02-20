import asyncio
import pandas as pd
import os
import time
from spade import agent, behaviour
from spade.message import Message
import json

SEND_INTERVAL = 1

class ConsumptionSenderAgent(agent.Agent):
    class SendMessageBehaviour(behaviour.CyclicBehaviour):
        def __init__(self, directory):
            super().__init__()
            script_dir = os.path.dirname(__file__)
            self.directory = os.path.abspath(os.path.join(script_dir, directory))
            print(f"Directory: {self.directory}")
            self.data = self.load_all_excel_files(self.directory)
            self.indices = {sheet: 0 for sheet in self.data.keys()}

        def load_all_excel_files(self, directory):
            data = {}
            for filename in os.listdir(directory):
                if filename.startswith("households_consumption_data_") and filename.endswith(".xlsx"):
                    file_path = os.path.join(directory, filename)
                    print(f"Loading file: {file_path}")
                    file_data = pd.read_excel(file_path, sheet_name=None)
                    for sheet_name, df in file_data.items():
                        if sheet_name in data:
                            data[sheet_name] = pd.concat([data[sheet_name], df], ignore_index=True)
                        else:
                            data[sheet_name] = df
            return data

        async def run(self):
            if self.agent.client is None:
                print("Disconnected")
                return

            all_sent = True
            for sheet, df in self.data.items():
                if self.indices[sheet] < len(df):
                    row = df.iloc[self.indices[sheet]]
                    timestamp = int(time.mktime(row.iloc[0].timetuple()))
                    message_content = {
                        "type": "consumption",
                        "timestamp": timestamp,
                        "household_id": sheet.split('_')[-1],
                        "consumption": row.iloc[1]
                    }
                    msg = Message(to="bluecoc@xmpp.is")
                    msg.body = json.dumps(message_content)
                    await self.send(msg)
                    print(f"Message Sent: {message_content}")
                    self.indices[sheet] += 1
                    all_sent = False

            if all_sent:
                print("All rows sent for all households")
                self.kill()

            await asyncio.sleep(SEND_INTERVAL)

    async def setup(self):
        print("ConsumptionSenderAgent launched")
        self.add_behaviour(self.SendMessageBehaviour("../../../data/output/consumption"))

async def main():
    sender = ConsumptionSenderAgent("bluecoc@xmpp.is", "fxa4VMG-nej7bpk3ycu")
    await sender.start()

    await asyncio.sleep(99999999999)
    await sender.stop()

asyncio.run(main())