import asyncio
import json
from spade import agent, behaviour
from spade.message import Message
from aiohttp import web

# 配置
households = {str(i): {"remaining": 100, "status": "1", "behavior": [], "high_energy_duration": 0} for i in range(1, 101)}
trade_records = []  # 存储交易记录

class FacilitingAgent(agent.Agent):
    class ReceiveConsumptionBehaviour(behaviour.CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=1)
            if msg:
                try:
                    data = json.loads(msg.body)
                    household_id = data.get("household_id")

                    if household_id not in households:
                        print(f"Unknown household ID: {household_id}")
                        return

                    message_type = data.get("type")
                    if message_type == "consumption":
                        consumption = float(data["consumption"])
                        households[household_id]["remaining"] -= consumption
                        if households[household_id]["remaining"] <= 0:
                            households[household_id]["remaining"] = 0
                            households[household_id]["status"] = "0"
                        print(f"Updated {household_id}: {households[household_id]}")

                    elif message_type == "production":
                        production = float(data["production"])
                        households[household_id]["remaining"] += production
                        if households[household_id]["remaining"] > 0:
                            households[household_id]["status"] = "1"
                        print(f"Updated {household_id}: {households[household_id]}")

                    elif message_type == "behavior":
                        households[household_id]["behavior"].append(data)

                    else:
                        print(f"Received unknown message type: {message_type}")

                except Exception as e:
                    print(f"Error processing message: {e}")

    class MonitorHouseholdsBehaviour(behaviour.PeriodicBehaviour):
        async def run(self):
            for household_id, data in households.items():
                remaining = data["remaining"]
                if remaining < 20:
                    print(f"Household {household_id} has low energy ({remaining}°). Requesting to buy electricity.")
                    await self.request_energy_trade(household_id, "buy", amount=20)
                elif remaining > 80:
                    data["high_energy_duration"] += 1
                    if data["high_energy_duration"] > 10:
                        print(f"Household {household_id} has high energy ({remaining}°) for a long time. Requesting to sell electricity.")
                        await self.request_energy_trade(household_id, "sell", amount=20)
                else:
                    data["high_energy_duration"] = 0

        async def request_energy_trade(self, household_id, trade_type, amount):
            msg = Message(to="negotiation_agent@xmpp.is")
            msg.set_metadata("performative", "request")
            msg.body = json.dumps({"household_id": household_id, "type": trade_type, "amount": amount, "timestamp": str(self.agent.now())})
            await self.send(msg)

    class ReceiveTradeDataBehaviour(behaviour.CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=1)
            if msg:
                try:
                    data = json.loads(msg.body)
                    trade_info = data.get("trade")
                    prices = data.get("prices")

                    if trade_info:
                        trade_records.append(trade_info)
                        print(f"Recorded trade: {trade_info}")

                    if prices:
                        print(f"Updated swap prices: {prices}")

                except Exception as e:
                    print(f"Error processing trade data: {e}")

    class HttpServerBehaviour(behaviour.OneShotBehaviour):
        async def run(self):
            async def handle_get_households(request):
                return web.json_response(households)

            async def handle_get_trades(request):
                return web.json_response(trade_records)

            app = web.Application()
            app.router.add_get('/households', handle_get_households)
            app.router.add_get('/trades', handle_get_trades)

            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, 'localhost', 8080)
            await site.start()

            print("HTTP server started at http://localhost:8080/")
            
            while True:
                await asyncio.sleep(3600)

    async def setup(self):
        print("FacilitingAgent started, monitoring households and processing data...")
        self.add_behaviour(self.ReceiveConsumptionBehaviour())
        self.add_behaviour(self.HttpServerBehaviour())
        self.add_behaviour(self.MonitorHouseholdsBehaviour(period=5))
        self.add_behaviour(self.ReceiveTradeDataBehaviour())

async def main():
    faciliting = FacilitingAgent("wxu20@xmpp.is", "@112233445566")
    await faciliting.start()
    await asyncio.sleep(99999999999)
    await faciliting.stop()

asyncio.run(main())