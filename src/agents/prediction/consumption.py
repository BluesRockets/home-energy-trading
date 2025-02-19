import asyncio
import pandas as pd
import os
from spade import agent, behaviour
from spade.message import Message

class ConsumptionSenderAgent(agent.Agent):
    class SendMessageBehaviour(behaviour.CyclicBehaviour):
        def __init__(self, excel_file):
            super().__init__()
            script_dir = os.path.dirname(__file__)
            self.excel_file = os.path.join(script_dir, excel_file)
            self.data = pd.read_excel(self.excel_file)
            self.index = 0

        async def run(self):
            if self.agent.client is None:
                print("Disconnected")
                return

            if self.index < len(self.data):
                row = self.data.iloc[self.index].to_dict()
                msg = Message(to="bluecoc@xmpp.is")
                msg.body = str(row)
                await self.send(msg)
                print(f"Message Sent: {row}")
                self.index += 1
            else:
                print("All rows sent")
                self.kill()

            await asyncio.sleep(1)

    async def setup(self):
        print("ConsumptionSenderAgent launched")
        self.add_behaviour(self.SendMessageBehaviour("../../../data/output/consumption/households_consumption_data_1.xlsx"))

async def main():
    sender = ConsumptionSenderAgent("bluecoc@xmpp.is", "fxa4VMG-nej7bpk3ycu")
    await sender.start()

    await asyncio.sleep(99999999999)
    await sender.stop()

asyncio.run(main())