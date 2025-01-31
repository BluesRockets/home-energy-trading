import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def read_data():
    file_path = '../../../data/input/power_grid_energy_consumption_dataset.xlsx'
    sheet_name = 'UsageData'

    df = pd.read_excel(file_path, sheet_name=sheet_name)

    return df

def real_data_analysis():
    df_real = pd.read_excel("../../../data/input/power_grid_energy_consumption_dataset.xlsx", sheet_name = 'UsageData')

    print(df_real.describe())

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # daily cycle
    df_real.groupby(df_real.index.hour).mean().plot(ax=axes[0], title="average daily load curve")

    # weekly cycle
    df_real.groupby([df_real.index.dayofweek, df_real.index.hour]).mean().unstack(0).plot(ax=axes[1], title="weekend and weekday difference")

    # monthly cycle
    df_real.resample("M").mean().plot(ax=axes[2], title="seasonal cycle")

def draw_plot_by_day(df):
    plt.figure(figsize=(10, 6))
    for i in range(1, 10):
        plt.plot(np.arange(24), df.iloc[i, 1:25].values, label=str(i) + "day (kWh)", marker='o', linestyle='-',
                 linewidth=2)
    plt.title("Energy Consumption Chart Per hours of a family", fontsize=16)
    plt.xlabel("hours", fontsize=12)
    plt.ylabel("consumption(kWh)", fontsize=12)
    plt.xticks(np.arange(24), fontsize=10)
    plt.grid(alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

def draw_plot_by_month(df):
    plt.figure(figsize=(10, 6))
    total_consumption = []
    for i in range(0, 12):
        consumption_daily = 0
        for item in df.iloc[30 * i, 1:25].values:
            consumption_daily += item
        total_consumption.append(consumption_daily)
    print(total_consumption)
    plt.plot(np.arange(12), total_consumption, label=str(i) + "day (kWh)", marker='o', linestyle='-',
                 linewidth=2)
    plt.title("Energy Consumption Chart a day in 12 months", fontsize=16)
    plt.xlabel("month", fontsize=12)
    plt.ylabel("consumption(kWh)", fontsize=12)
    plt.xticks(np.arange(12), fontsize=10)
    plt.grid(alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


def main():
    df = read_data()
    draw_plot_by_day(df)
    draw_plot_by_month(df)
    real_data_analysis()




if __name__ == "__main__":
    main()
