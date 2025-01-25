import os
import pandas as pd
import numpy as np

# File paths
file_path = '../../data/input/power_grid_energy_consumption_dataset.xlsx'
output_file_path = '../../data/output/consumption/households_energy_consumption_dataset.xlsx'

# Check if the input file exists
if not os.path.exists(file_path):
    current_directory = os.getcwd()
    raise FileNotFoundError(
        f"The file '{file_path}' does not exist. Please ensure the file is in the correct path.\n"
        f"Current directory: {current_directory}\n"
        f"Expected location: {os.path.join(current_directory, file_path)}"
    )

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

# Ensure output directory exists before saving
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

# Save to a new Excel file
all_households_df.to_excel(output_file_path, index=False)

print(f"The data has been successfully saved to '{output_file_path}'.")