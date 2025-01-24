import pandas as pd

def read_data():
    file_path = '../../../data/input/power_grid_energy_consumption_dataset.xlsx'
    sheet_name = 'UsageData'

    df = pd.read_excel(file_path, sheet_name=sheet_name)

    print(df.head())

def main():
    read_data()


if __name__ == "__main__":
    main()
