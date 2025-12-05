import os
import matplotlib.pyplot as plt
import seaborn as sns

pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def plot_boxplots(df, output_dir):
    box_dir = os.path.join(output_dir, "boxplots")
    ensure_dir(box_dir)

    print("\n=== PLOT: BOXPLOTS ===")

    for col in pollutants:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col])
        plt.title(f"{col} Boxplot (Outliers)")

        save_path = os.path.join(box_dir, f"{col}_boxplot.png")
        plt.savefig(save_path)
        plt.close()

        print(f"Saved: {save_path}")

    print("Boxplot generation completed.\n")
