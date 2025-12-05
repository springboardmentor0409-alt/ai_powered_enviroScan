import os
import matplotlib.pyplot as plt
import pandas as pd

pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def plot_time_trends(df, output_dir):
    ensure_dir(output_dir)

    print("\n=== PLOT: DAILY TIME TRENDS ===")

    # Ensure date is a datetime type
    df['date'] = pd.to_datetime(df['date'])

    # Daily average pollutant levels
    daily = df.groupby("date")[pollutants].mean()

    # Plot
    plt.figure(figsize=(12, 6))
    for col in pollutants:
        plt.plot(daily.index, daily[col], label=col)

    plt.title("Daily Average Pollutant Trends")
    plt.xlabel("Date")
    plt.ylabel("Pollution Level")
    plt.legend()

    save_path = os.path.join(output_dir, "time_trends.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Saved: {save_path}")
    print("Time trend plot completed.\n")
