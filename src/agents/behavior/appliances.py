
import asyncio
import pandas as pd
import os
import time
from spade import agent, behaviour
from spade.message import Message
import json
from datetime import datetime, timedelta

# Set interval between messages in seconds
SEND_INTERVAL = 1


class HouseholdConsumptionAgent(agent.Agent):
    class SendDataBehaviour(behaviour.CyclicBehaviour):
        def __init__(self):
            super().__init__()
            # Load data files from the current directory
            self.data = self.load_household_data()
            # Track current hour index for sending data
            self.current_hour_index = 0
            # Get unique hours from the dataset
            if self.data is not None:
                self.unique_hours = sorted(self.data['time'].unique())
                self.household_ids = sorted(self.data['household_id'].unique())
                # Initialize household index
                self.current_household_index = 0
            else:
                self.unique_hours = []
                self.household_ids = []
                print("Error: No data loaded")

            # Flag to indicate if we're done
            self.all_data_sent = False

        def load_household_data(self):
            try:
                # Read CSV files directly - now loading all three files
                script_dir = os.path.dirname(__file__)
                df1 = pd.read_csv(os.path.abspath(os.path.join(script_dir, '../../../data/output/appliances/household_data1.csv')))
                df2 = pd.read_csv(os.path.abspath(os.path.join(script_dir, '../../../data/output/appliances/household_data2.csv')))
                df3 = pd.read_csv(os.path.abspath(os.path.join(script_dir, '../../../data/output/appliances/household_data3.csv')))

                # Ensure time column is in datetime format
                df1['time'] = pd.to_datetime(df1['time'])
                df2['time'] = pd.to_datetime(df2['time'])
                df3['time'] = pd.to_datetime(df3['time'])

                # Merge data
                combined_df = pd.concat([df1, df2, df3], ignore_index=True)
                return combined_df

            except FileNotFoundError as e:
                print(f"Error: Could not find file - {e}")
                return None
            except Exception as e:
                print(f"Error loading data: {str(e)}")
                return None

        def get_future_hours_data(self, timestamp, household_id, appliance, hours=23):
            """从CSV数据中获取未来指定小时数的预测数据"""
            future_data = []

            # 获取当前时间之后的23个小时
            future_times = [timestamp + timedelta(hours=i + 1) for i in range(hours)]

            # 为每个未来时间点找到对应的数据
            for future_time in future_times:
                future_row = self.data[(self.data['time'] == future_time) &
                                       (self.data['household_id'] == household_id)]

                if not future_row.empty and appliance in future_row.columns:
                    # 获取该时间点的设备数据
                    value = round(float(future_row[appliance].iloc[0]), 3)
                    future_data.append(value)
                else:
                    # 如果找不到数据，使用0作为占位符
                    future_data.append(0.0)

            return future_data

        def format_household_message(self, timestamp, household_id, row):
            """Format a household's data into JSON message with future predictions"""
            # Get appliance columns
            appliance_columns = [col for col in self.data.columns if col not in ['time', 'household_id']]

            # Create JSON structure
            message_dict = {
                "type": "behavior",
                "timestamp": timestamp.strftime('%Y-%m-%dT%H:%M:%S') + 'Z',
                "household_id": household_id,
                "data": {}
            }

            # Add current and future appliance readings
            for appliance in appliance_columns:
                if appliance in row and not pd.isna(row[appliance]):
                    current_value = round(float(row[appliance]), 3)
                    # 从CSV数据中获取未来23小时的预测值
                    future_values = self.get_future_hours_data(timestamp, household_id, appliance)
                    # 当前值加上未来23小时的预测值
                    message_dict["data"][appliance] = [current_value] + future_values

            # Convert to JSON string
            return json.dumps(message_dict)

        async def run(self):
            # Check if we have data and if we haven't sent all hours yet
            if self.data is None or self.all_data_sent:
                self.kill()
                return

            # Get current hour to process
            current_hour = self.unique_hours[self.current_hour_index]
            current_household = self.household_ids[self.current_household_index]

            # Filter data for this hour and household
            hour_data = self.data[(self.data['time'] == current_hour) &
                                  (self.data['household_id'] == current_household)]

            if not hour_data.empty:
                # Format message with all appliances for this household
                message_text = self.format_household_message(
                    current_hour, current_household, hour_data.iloc[0])

                # Create and send the message
                msg = Message(to="wxu20@xmpp.is")  # Recipient
                msg.body = message_text
                await self.send(msg)

                # Print the message in console
                print(f"{message_text}")

            # Move to next household
            self.current_household_index += 1

            # If we've processed all households for this hour, move to next hour
            if self.current_household_index >= len(self.household_ids):
                self.current_household_index = 0
                self.current_hour_index += 1

            # Check if we've processed all hours
            if self.current_hour_index >= len(self.unique_hours):
                self.all_data_sent = True

            # Wait before sending next message
            await asyncio.sleep(SEND_INTERVAL)

    async def setup(self):
        send_behavior = self.SendDataBehaviour()
        self.add_behaviour(send_behavior)


async def main():
    # Create agent with credentials
    sender = HouseholdConsumptionAgent("appliance@xmpp.is", "prediction5014")

    # Start the agent
    await sender.start()

    try:
        # Keep the agent running until all data is sent
        while not sender.behaviours[0].is_killed():
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop the agent
        await sender.stop()


if __name__ == "__main__":
    asyncio.run(main())

