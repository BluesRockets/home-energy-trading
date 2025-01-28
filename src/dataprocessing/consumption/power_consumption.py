import os
import pandas as pd
import numpy as np

# File paths
file_path = '../../data/input/power_grid_energy_consumption_dataset.xlsx'
output_file_path_1 = '../../data/output/consumption/households_energy_consumption_dataset_part1.xlsx'
output_file_path_2 = '../../data/output/consumption/households_energy_consumption_dataset_part2.xlsx'

# Check if the input file exists
if not os.path.exists(file_path):
    current_directory = os.getcwd()
    raise FileNotFoundError(
        f"The file '{file_path}' does not exist. Please ensure the file is in the correct path.\n"
        f"Current directory: {current_directory}\n"
        f"Expected location: {os.path.join(current_directory, file_path)}"
    )

# 12345

# Read the Excel file
df = pd.read_excel(file_path)

# Generate load data for 100 households
num_households = 100
household_data = []

for i in range(num_households):
    # Generate load data for each household (random perturbation of original data)
    household_load = df.copy()
    # Apply random variation to data, skipping first column
    household_load.iloc[:, 1:] = household_load.iloc[:, 1:] * (
            1 + np.random.normal(0, 0.1, household_load.iloc[:, 1:].shape)
    )
    household_load['Household'] = f'Household_{i + 1}'
    household_data.append(household_load)

# Combine all households' load data
all_households_df = pd.concat(household_data, ignore_index=True)

# Split the data into two parts
mid_index = len(all_households_df) // 2
part1_df = all_households_df.iloc[:mid_index]
part2_df = all_households_df.iloc[mid_index:]

# Ensure output directory exists before saving
os.makedirs(os.path.dirname(output_file_path_1), exist_ok=True)

# Save to new Excel files
part1_df.to_excel(output_file_path_1, index=False)
part2_df.to_excel(output_file_path_2, index=False)

print(f"The data has been successfully saved to '{output_file_path_1}' and '{output_file_path_2}'.")