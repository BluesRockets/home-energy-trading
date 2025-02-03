import numpy as np
import pandas as pd
from scipy.stats import skewnorm, truncnorm
import matplotlib.pyplot as plt

NUMBER_OF_FAMILIES = 1


def read_data():
    file_path = '../../../data/input/power_grid_energy_consumption_dataset.xlsx'
    sheet_name = 'UsageData'

    df = pd.read_excel(file_path, sheet_name=sheet_name)

    return df


def get_consumption_level_factor():
    mean = 1.0
    std_dev = 0.3
    lower_bound, upper_bound = 0.6, 2.5

    a = (lower_bound - mean) / std_dev
    b = (upper_bound - mean) / std_dev

    return truncnorm.rvs(a, b, loc=mean, scale=std_dev)


def get_family_member_factor():
    mean = 1.0
    std_dev = 0.3
    lower_bound, upper_bound = 0.6, 2

    a = (lower_bound - mean) / std_dev
    b = (upper_bound - mean) / std_dev

    return truncnorm.rvs(a, b, loc=mean, scale=std_dev)


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
        num = np.random.randint(1, 100)
        if num <= 10:
            return np.random.randint(9, 12) / 10
        elif num <= 40:
            return np.random.randint(12, 15) / 10
        else:
            return np.random.randint(15, 20) / 10
    elif month == 1 or month == 3:  # spring&aut
        num = np.random.randint(1, 100)
        if num <= 15:
            return np.random.randint(7, 10) / 10
        elif num <= 15:
            return np.random.randint(13, 16) / 10
        else:
            return np.random.randint(10, 13) / 10
    else:  # summer
        num = np.random.randint(1, 100)
        if num <= 20:
            return np.random.randint(3, 5) / 10
        elif num <= 40:
            return np.random.randint(7, 10) / 10
        else:
            return np.random.randint(5, 7) / 10


def generate_household(hh_id, year=2024):
    index = pd.date_range(f"{year}-01-01", f"{year}-1-3 23:00", freq="h")

    df = pd.DataFrame(index=index)
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["is_weekend"] = df.day_of_week >= 5
    df["season"] = df.index.month.map({12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3})

    # household config
    params = {
        "base_load": 0.85,
        "consumption_level": get_consumption_level_factor(),
        "family_member": get_family_member_factor(),
        "weekend_multiplier": 1 + np.random.uniform(-1, 1) * 0.3,
        "peak_factor": get_peak_factor()
    }

    df["daily_cycle"] = df["hour"].apply(get_daily_hour_factor)

    df["load"] = params["base_load"]

    # family member & consumption level factor
    df["load"] += params["family_member"] * params["consumption_level"]

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
    noise = skewnorm.rvs(5, loc=0, scale=0.1, size=len(df))
    df["load"] = np.abs(df["load"] + noise)
    df["hh_id"] = hh_id

    return df


def generate_multiple_households(num_households):
    frames = []
    for hh_id in range(1, num_households + 1):
        household_df = generate_household(hh_id)
        frames.append(household_df)

    combined_df = pd.concat(frames, ignore_index=True)
    return combined_df


def plot_multiple_households(combined_df):
    plt.figure(figsize=(12, 6))

    for hh_id, household_data in combined_df.groupby("hh_id"):
        plt.plot(household_data["hour"],
                 household_data["load"],
                 marker='o', label=f"HH {hh_id}")

    plt.title("daily usage for multiple households", fontsize=16)
    plt.xlabel("Hour", fontsize=12)
    plt.ylabel("daily usage", fontsize=12)
    plt.legend(title="Households", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.show()


def main():
    combined_df = generate_multiple_households(NUMBER_OF_FAMILIES)

    plot_multiple_households(combined_df)


if __name__ == "__main__":
    main()
