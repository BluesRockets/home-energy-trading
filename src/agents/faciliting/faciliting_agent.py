import asyncio
import json
from spade import agent, behaviour
from spade.message import Message

# config
households = {str(i): {"remaining": 100, "status": "1", "behavior": []} for i in range(1, 101)}


class FacilitingAgent(agent.Agent):
    class ReceiveConsumptionBehaviour(behaviour.CyclicBehaviour):
        async def run(self):
            # waiting for message timeout 1s
            msg = await self.receive(timeout=1)
            if msg:
                try:
                    data = json.loads(msg.body)
                    household_id = data.get("household_id")

                    if household_id not in households:
                        print(f"Unknown household ID: {household_id}")
                        return

                    match data.get("type"):
                        case "consumption":
                            consumption = float(data["consumption"])
                            households[household_id]["remaining"] -= consumption
                            if households[household_id]["remaining"] <= 0:
                                households[household_id]["remaining"] = 0
                                households[household_id]["status"] = "0"
                            print(f"Updated {household_id}: {households[household_id]}")

                        case "production":
                            production = float(data["production"])
                            households[household_id]["remaining"] += production
                            if households[household_id]["remaining"] > 0:
                                households[household_id]["status"] = "1"
                            print(f"Updated {household_id}: {households[household_id]}")

                        case "behavior":
                            #这里可能需要修改
                            households[household_id]["behavior"].append(data)

                        case _:
                            print(f"Received unknown message type: {data.get('type')}")

                except Exception as e:
                    print(f"Error processing message: {e}")

    async def setup(self):
        print("FacilitingAgent started, listening for consumption and production data...")
        self.add_behaviour(self.ReceiveConsumptionBehaviour())


async def main():
    faciliting = FacilitingAgent("wxu20@xmpp.is", "@11223445566")
    await faciliting.start()
    await asyncio.sleep(99999999999)
    await faciliting.stop()


asyncio.run(main())
