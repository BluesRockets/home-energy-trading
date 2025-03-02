import asyncio
import json
from spade import agent, behaviour
from spade.message import Message

# config
households = {str(i): {"remaining": 100, "status": "1"} for i in range(1, 101)}


class FacilitingAgent(agent.Agent):
    class ReceiveConsumptionBehaviour(behaviour.CyclicBehaviour):
        async def run(self):
            # waiting for message timeout 10s
            msg = await self.receive(timeout=10)
            if msg:
                try:
                    data = json.loads(msg.body)

                    # if message type="consumption"
                    if data.get("type") == "consumption":
                        household_id = data["household_id"]
                        # get consumption value
                        consumption = float(data["consumption"])

                        if household_id in households:
                            # minus consumption data
                            households[household_id]["remaining"] -= consumption
                            if households[household_id]["remaining"] <= 0:
                                households[household_id]["remaining"] = 0
                                # battery is empty
                                households[household_id]["status"] = "0"
                            print(f"Updated {household_id}: {households[household_id]}")
                        else:
                            print(f"Unknown household ID: {household_id}")

                    else:
                        print(f"Received unknown message type: {data.get('type')}")

                except Exception as e:
                    print(f"Error processing message: {e}")

    async def setup(self):
        print("FacilitingAgent started, listening for consumption data...")
        self.add_behaviour(self.ReceiveConsumptionBehaviour())


async def main():
    faciliting = FacilitingAgent("faciliting@xmpp.is", "your_password")
    await faciliting.start()

    await asyncio.sleep(99999999999)
    await faciliting.stop()


asyncio.run(main())