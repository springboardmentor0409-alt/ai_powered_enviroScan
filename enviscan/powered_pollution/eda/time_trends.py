import matplotlib.pyplot as plt
import pandas as pd

pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]

def plot_time_trends(df):
    df['date'] = pd.to_datetime(df['date'])
    daily = df.groupby("date")[pollutants].mean()

    daily.plot(figsize=(12, 6))
    plt.title("Daily Average Pollutant Trends")
    plt.xlabel("Date")
    plt.ylabel("Level")
    plt.show()
