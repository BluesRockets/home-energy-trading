import numpy as np
import pandas as pd
from scipy.stats import skewnorm

import matplotlib.pyplot as plt

NUMBER_OF_FAMILIES = 1

def read_data():
    file_path = '../../../data/input/power_grid_energy_consumption_dataset.xlsx'
    sheet_name = 'UsageData'

    df = pd.read_excel(file_path, sheet_name=sheet_name)

    return df

def get_consumption_level_factor():
    num = np.random.randint(1, 100)
    if num <= 20:
        return np.random.randint(6, 9) / 10
    elif num <=60:
        return np.random.randint(9, 11) / 10
    else:
        return np.random.randint(11, 18) / 10

def get_family_member_factor():
    num = np.random.randint(1, 100)
    if num <= 20:
        return np.random.randint(6, 9) / 10
    elif num <=60:
        return np.random.randint(9, 11) / 10
    else:
        return np.random.randint(11, 18) / 10

#获得当前所处小时时间后进行概率分部,以3kwh为基准1，注意！！！此处为单户家庭的每日用电习惯的概率分布，并非不同家庭的用电习惯
def get_daily_hour_factor(hour):
    # 用电高峰15-21，概率分部为0.7-0.9 10%；0.9-2.1：20%；2.1-3.5:70%
    if 15 <= hour < 21:
        num = np.random.randint(1, 100)
        if num <= 10:
            return np.random.randint(7, 9) / 10
        elif num <= 30:
            return np.random.randint(9, 21) / 10
        else:
            return np.random.randint(21, 35) / 10
    # 低峰2-10 概率分部为0.4-0.6：10%；0.9-1.1：10%；0.6-0.9:80%
    elif 2<=hour<10:
        num = np.random.randint(1, 100)
        if num <= 10:
            return np.random.randint(4, 6) / 10
        elif num <= 20:
            return np.random.randint(9, 11) / 10
        else:
            return np.random.randint(6, 9) / 10
        # 其他时间
    else:
        num = np.random.randint(1, 100)
        if num <= 20:
            return np.random.randint(7, 9) / 10
        elif num <= 70:
            return np.random.randint(15, 25) / 10
        else:
            return np.random.randint(9, 15) / 10

def get_season_factor(month):
    if month in [0,1,10, 11]:  # winter
        return np.random.uniform(0.7, 2)
    elif month in [2,3, 4, 5]:  # spring
        return np.random.uniform(0.7, 1.5)
    else:  # summer6-8
        return np.random.uniform(0.2, 0.7)  # 夏季用电较少

# def get_season_factor(month):
#     if 0<=month<=5 or 10<=month<=11:
#         return np.random.randint(2.5, 9) / 10
#     else:
#         return np.random.randint(9, 18) / 10


# def get_season_factor(hour):
#
#     return


def generate_household(hh_id, year=2024):
    index = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
    df = pd.DataFrame(index=index)
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["is_weekend"] = df.day_of_week >= 5
    df["season"] = df.index.month.map({12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3})

    # household config
    params = {
        "base_load": 1,
        "consumption_level": get_consumption_level_factor(),
        "family_member": get_family_member_factor(),
        "weekend_multiplier": 1 + np.random.uniform(-1, 1) * 0.3,
        "summer_boost": np.random.uniform(1.2, 1.8) if np.random.rand() < 0.7 else 1.0,

        "daily_boost": df["hour"].apply(get_daily_hour_factor)

        #"season_boost": get_season_factor(df["season"]),
    }
    df["load"] = params["base_load"]
    # daily cycle
    df["load"] += params["family_member"] * params["consumption_level"]

    # weekend boost
    df["load"] *= np.where(df["is_weekend"], params["weekend_multiplier"], 1)

    # Apply hourly factor for daily cycle fluctuation
    # df["load"] *= df["hour"].apply(get_daily_hour_factor) * (
    #         np.sin(df["hour"] / 24 * 2 * np.pi) * 0.5 + 0.5
    # )
    # Apply seasonal factor based on season data
    # df["load"] *= df["season"].map(get_season_factor)

    # add noise
    noise = skewnorm.rvs(5, loc=0, scale=0.1, size=len(df))
    df["load"] = np.abs(df["load"] + noise)

    return df


def main():
    for i in range(NUMBER_OF_FAMILIES):
        generate_household(i)[["load"]]


if __name__ == "__main__":
    main()

