import json
import asyncio
import datetime
from spade import agent, behaviour
from spade.message import Message
from contract import swap_dollar_to_electricity, swap_electricity_to_dollar, get_swap_price, w3  # 导入交易接口

FACILITATING_AGENT_JID = "faciliting_agent@xmpp.is"

class NegotiationAgent(agent.Agent):
    class HandleTradeRequestsBehaviour(behaviour.CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=1)
            if msg:
                data = json.loads(msg.body)
                household_id = data["household_id"]
                trade_type = data["trade_type"]
                amount = data["amount"]

                print(f"Received trade request from household {household_id}: {trade_type} {amount} units.")

                price_a_to_b, price_b_to_a = get_swap_price()  # 获取最新价格信息

                trade_data = {
                    "household_id": household_id,
                    "trade_type": trade_type,
                    "amount_in": amount,
                    "amount_out": None,
                    "tx_hash": None,
                    "timestamp": datetime.datetime.utcnow().isoformat()
                }

                if trade_type == "buy":
                    dollar_in = w3.to_wei(amount, 'ether')
                    min_electricity_out = amount  # 允许最小的电量交换
                    receipt = swap_dollar_to_electricity(dollar_in, min_electricity_out)

                    trade_data["amount_out"] = min_electricity_out
                    trade_data["tx_hash"] = receipt.transactionHash.hex()

                    print(f"Buy electricity transaction receipt: {trade_data['tx_hash']}")

                elif trade_type == "sell":
                    electricity_in = w3.to_wei(amount, 'ether')
                    min_dollar_out = amount  # 允许最小的钱交换
                    receipt = swap_electricity_to_dollar(electricity_in, min_dollar_out)

                    trade_data["amount_out"] = min_dollar_out
                    trade_data["tx_hash"] = receipt.transactionHash.hex()

                    print(f"Sell electricity transaction receipt: {trade_data['tx_hash']}")

                # 发送交易数据到 FacilitatingAgent
                await self.send_trade_info(trade_data, price_a_to_b, price_b_to_a)

        async def send_trade_info(self, trade_data, price_a_to_b, price_b_to_a):
            """发送交易信息和实时价格到 FacilitatingAgent"""
            message_body = {
                "trade": trade_data,
                "prices": {
                    "price_a_to_b": price_a_to_b,
                    "price_b_to_a": price_b_to_a
                }
            }

            msg = Message(to=FACILITATING_AGENT_JID)
            msg.set_metadata("performative", "inform")
            msg.body = json.dumps(message_body)
            await self.send(msg)
            print("Sent trade info to FacilitingAgent:", message_body)

    async def setup(self):
        print("NegotiationAgent started. Waiting for trade requests...")
        self.add_behaviour(self.HandleTradeRequestsBehaviour())

async def main():
    negotiation = NegotiationAgent("negotiation_agent@xmpp.is", "password")
    await negotiation.start()
    await asyncio.sleep(99999999999)
    await negotiation.stop()

asyncio.run(main())