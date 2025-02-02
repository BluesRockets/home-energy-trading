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
    elif num <= 60:
        return np.random.randint(9, 11) / 10
    else:
        return np.random.randint(11, 18) / 10


def get_family_member_factor():
    num = np.random.randint(1, 100)
    if num <= 20:
        return np.random.randint(6, 9) / 10
    elif num <= 60:
        return np.random.randint(9, 11) / 10
    else:
        return np.random.randint(11, 18) / 10

def get_peak_factor():
    num = np.random.randint(1, 100)
    if num <= 10:
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
    index = pd.date_range(f"{year}-01-01", f"{year}-01-01 23:00", freq="h")
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

        "peak_factor":get_peak_factor()
        # "summer_boost": np.random.uniform(1.2, 1.8) if np.random.rand() < 0.7 else 1.0,
        # 非矢量操作效率低，当前仅仅测试可优化

        # "season_boost": get_season_factor(df["season"]),
    }

    df["daily_cycle"] = df["hour"].apply(get_daily_hour_factor)
    # print(df)
    df["load"] = params["base_load"]
    # daily cycle
    df["load"] += params["family_member"] * params["consumption_level"]

    # weekend boost
    df["load"] *= np.where(df["is_weekend"], params["weekend_multiplier"], 1)

    # Apply hourly factor for daily cycle fluctuation
    # 适合参数相位10，周期22；相位12周期28
    df["load"] *= (
        # A取0.45，y轴位移1.1（目前最佳）或0.5,1.2
            np.sin(2 * np.pi * (df["hour"] - params["peak_factor"]) / 28) * 0.45 + 1.1
    )
    # Apply seasonal factor based on season data
    df["season_cycle"] = df["season"].apply(get_season_factor)
    df["load"] *= df["season_cycle"]
    # add noise
    # noise = skewnorm.rvs(5, loc=0, scale=0.1, size=len(df))
    # df["load"] = np.abs(df["load"] + noise)
    df["hh_id"] = hh_id

    return df


# def main():
#     for i in range(NUMBER_OF_FAMILIES):
#         generate_household(i)[["load"]]

# 生成多个家庭数据
def generate_multiple_households(num_households):
    frames = []
    for hh_id in range(1, num_households + 1):
        household_df = generate_household(hh_id)
        frames.append(household_df)

    combined_df = pd.concat(frames, ignore_index=True)
    return combined_df


# 绘制图表：将多个家庭的数据绘制在同一个图上
def plot_multiple_households(combined_df):
    plt.figure(figsize=(12, 6))  # 设置图表大小

    # 按照家庭 ID (hh_id) 分组绘制
    for hh_id, household_data in combined_df.groupby("hh_id"):
        plt.plot(household_data["hour"],
                 household_data["load"],
                 marker='o', label=f"HH {hh_id}")  # 每个家庭一条折线

    # 设置标题、坐标轴标签、图例和网格
    plt.title("daily_cycle for Multiple Households", fontsize=16)
    plt.xlabel("Hour", fontsize=12)
    plt.ylabel("daily_cycle", fontsize=12)
    plt.legend(title="Households", fontsize=10)  # 显示家庭 ID 的图例
    plt.grid(True, linestyle="--", alpha=0.6)

    # 显示图表
    plt.show()


# 主函数
def main():
    # 生成多个家庭
    num_households = 8
    combined_df = generate_multiple_households(num_households)

    # 绘制所有家庭的 Daily Boost 在一个图上显示
    plot_multiple_households(combined_df)


if __name__ == "__main__":
    main()
