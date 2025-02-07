import numpy as np
import pandas as pd
from scipy.stats import skewnorm, truncnorm
import matplotlib.pyplot as plt
import random

NUMBER_OF_FAMILIES = 100


def read_data():
    file_path = '../../../data/input/power_grid_energy_consumption_dataset.xlsx'
    sheet_name = 'UsageData'

    df = pd.read_excel(file_path, sheet_name=sheet_name)

    return

def normal_distribution(mean, std_dev, lower_bound, upper_bound):

    a = (lower_bound - mean) / std_dev
    b = (upper_bound - mean) / std_dev

    return truncnorm.rvs(a, b, loc=mean, scale=std_dev)


def get_consumption_level_factor():
    return normal_distribution(1.0, 3, 0.6, 2.5)


def get_family_member_factor():
    return normal_distribution(3.0, 3, 1, 6)


def get_peak_factor():
    num = np.random.randint(1, 100)
    if num <= 20:
        return np.random.randint(-7, 3)
    else:
        return np.random.randint(10, 14)


def get_daily_hour_factor(hour):
    """
    Calculate the daily hour factor for energy consumption based on the hour of the day.
    Peak hours (15-21), off-peak hours (2-10), and other hours have varying probability distributions.
    """
    if 15 <= hour < 21:  # Peak hours
        num = np.random.randint(1, 100)
        if num <= 10:
            return np.random.randint(7, 9) / 10
        elif num <= 40:
            return np.random.randint(9, 12) / 10
        else:
            return np.random.randint(12, 20) / 10
    elif 2 <= hour < 10:  # Off-peak hours
        num = np.random.randint(1, 100)
        if num <= 10:
            return np.random.randint(4, 6) / 10
        elif num <= 30:
            return np.random.randint(9, 14) / 10
        else:
            return np.random.randint(6, 9) / 10
    else:  # Other hours
        num = np.random.randint(1, 100)
        if num <= 20:
            return np.random.randint(7, 9) / 10
        elif num <= 70:
            return np.random.randint(12, 15) / 10
        else:
            return np.random.randint(9, 12) / 10


def get_season_factor(month):
    if month == 0:  # winter
        return normal_distribution(2, 10, 1, 2.5)
    elif month == 1 or month == 3:  # spring&aut=
        return normal_distribution(1, 10, 0.5, 1.3)
    else:  # summer
        return normal_distribution(0.5, 10, 0.1, 0.7)


def assign_daily_parameters(row):
    params = {
        "base_load": normal_distribution(0.1, 3, 0.05, 0.2),
        "consumption_level": get_consumption_level_factor(),
        "family_member": get_family_member_factor(),
        "peak_factor": get_peak_factor()
    }
    return params['base_load'] * params["consumption_level"] * params["family_member"]


def generate_household(hh_id, year=2024):
    index = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")

    df = pd.DataFrame(index=index)
    df["hour"] = df.index.hour
    df["month"] = df.index.month
    df["day_of_week"] = df.index.dayofweek
    df["is_weekend"] = df.day_of_week >= 5
    df["season"] = df.index.month.map({12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3})

    # household config
    params = {
        "weekend_multiplier": 1 + np.random.uniform(-1, 1) * 0.3,
        "peak_factor": get_peak_factor()
    }

    df["daily_cycle"] = df["hour"].apply(get_daily_hour_factor)

    df["load"] = df["daily_cycle"]

    # weekend boost
    df["load"] *= np.where(df["is_weekend"], params["weekend_multiplier"], 1)

    # Apply hourly factor for daily cycle fluctuation
    df["load"] *= (
        np.sin(2 * np.pi * (df["hour"] - params["peak_factor"]) / 28) * 0.45 + 1.1
    )

    # Apply seasonal factor based on season data
    df["season_cycle"] = df["season"].apply(get_season_factor)
    df["load"] *= df["season_cycle"]

    # add noise
    noise = skewnorm.rvs(5, loc=0, scale=1, size=len(df['load']))
    df["load"] = np.abs(df["load"] * noise)
    df["hh_id"] = hh_id

    return df


def generate_multiple_households(num_households):
    frames = []
    for hh_id in range(1, num_households + 1):
        household_df = generate_household(hh_id)
        frames.append(household_df)

    combined_df = pd.concat(frames, ignore_index=False)
    combined_df.reset_index(inplace=True)
    combined_df.rename(columns={"index": "datetime"}, inplace=True)

    return combined_df


def plot_random_30_days_single_household(combined_df):
    random_hh_id = random.choice(combined_df["hh_id"].unique())

    household_data = combined_df[combined_df["hh_id"] == random_hh_id]

    if not pd.api.types.is_datetime64_any_dtype(household_data["datetime"]):
        household_data["datetime"] = pd.to_datetime(household_data["datetime"])

    household_data["day"] = household_data["datetime"].dt.date
    random_days = random.sample(list(household_data["day"].unique()), 30)

    household_random_30_days = household_data[household_data["day"].isin(random_days)]

    plt.figure(figsize=(12, 6))
    for day in sorted(household_random_30_days["day"].unique()):
        day_data = household_random_30_days[household_random_30_days["day"] == day]
        plt.plot(day_data["datetime"].dt.hour,
                 day_data["load"],
                 marker='o',
                 label=f"{day}")

    plt.title(f"Random 30 Days Energy Usage: Household {random_hh_id}", fontsize=16)
    plt.xlabel("Hour of the Day", fontsize=12)
    plt.ylabel("Hourly Consumption (kWh)", fontsize=12)
    plt.legend(title="Date", fontsize=10, loc="upper left", ncol=2)
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.show()


# total consumption per month by household
def plot_monthly_usage_by_household(combined_df):

    combined_df['datetime'] = pd.to_datetime(combined_df['datetime'])
    combined_df['month'] = combined_df['datetime'].dt.month

    monthly_usage = combined_df.groupby(['hh_id', 'month'])['load'].sum().unstack()

    plt.figure(figsize=(12, 6))

    for hh_id in monthly_usage.index:
        plt.plot(monthly_usage.columns,
                 monthly_usage.loc[hh_id],
                 marker='o', label=f"HH {hh_id}")

    plt.title("Monthly Energy Consumption by Household", fontsize=16)
    plt.xlabel("Month", fontsize=12)
    plt.xticks(ticks=range(1, 13), labels=[
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ])
    plt.ylabel("Monthly Energy Consumption (kWh)", fontsize=12)
    plt.legend(title="Households", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.show()

def export_households_to_excel(combined_df, base_file_name="household_energy_data", num_excel_files=3):
    family_ids = combined_df["hh_id"].unique()

    family_ids = combined_df["hh_id"].unique()
    families_per_file = len(family_ids) // num_excel_files

    for file_idx in range(num_excel_files):
        start_idx = file_idx * families_per_file
        if file_idx == num_excel_files - 1:
            end_idx = len(family_ids)
        else:
            end_idx = (file_idx + 1) * families_per_file

        current_family_ids = family_ids[start_idx:end_idx]

        excel_file_name = f"../../../data/output/consumption/{base_file_name}_{file_idx + 1}.xlsx"
        with pd.ExcelWriter(excel_file_name, engine="openpyxl") as writer:
            for hh_id in current_family_ids:
                household_data = combined_df[combined_df["hh_id"] == hh_id][["datetime", "load"]]
                household_data.set_index("datetime", inplace=True)
                household_data.to_excel(writer, sheet_name=f"Household_{hh_id}")

        print(f"Save Successfully：{excel_file_name}")


def main():
    combined_df = generate_multiple_households(NUMBER_OF_FAMILIES)

    # plot_multiple_households(combined_df)
    plot_random_30_days_single_household(combined_df)
    # plot_monthly_usage_by_household(combined_df)

    export_households_to_excel(combined_df, 'households_consumption_data')

if __name__ == "__main__":
    main()
