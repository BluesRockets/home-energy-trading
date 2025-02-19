import asyncio
from spade import agent, behaviour
from spade.message import Message

class demo_agent(agent.Agent):
    class send_message_behaviour(behaviour.CyclicBehaviour):
        async def run(self):
            if self.agent.client is None:
                print("❌ XMPP Disconnected")
                return

            msg = Message(to="bluecoc@xmpp.is")
            msg.body = "Hello, this is a test message."
            await self.send(msg)
            print("Message Sent")
            await asyncio.sleep(5)


    async def setup(self):
        print("SenderAgent launched")
        self.add_behaviour(self.send_message_behaviour())


class receiver_agent(agent.Agent):
    class receive_messages_behaviour(behaviour.CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=10)
            if msg:
                print(f"Message received: {msg.body}")

    async def setup(self):
        print("ReceiverAgent launched")
        self.add_behaviour(self.receive_messages_behaviour())

async def main():
    sender = demo_agent("bluecoc@xmpp.is", "fxa4VMG-nej7bpk3ycu")
    receiver = receiver_agent("bluecoc@xmpp.is", "fxa4VMG-nej7bpk3ycu")

    await receiver.start()
    await sender.start()

    await asyncio.sleep(99999999999)
    await sender.stop()
    await receiver.stop()

asyncio.run(main())