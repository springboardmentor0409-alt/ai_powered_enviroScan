import os
import matplotlib.pyplot as plt
import seaborn as sns

pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def plot_distributions(df, output_dir):
    # Create results/distributions/ folder
    dist_dir = os.path.join(output_dir, "distributions")
    ensure_dir(dist_dir)

    print("\n=== PLOT: DISTRIBUTIONS ===")

    for col in pollutants:
        plt.figure(figsize=(7, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f"{col} Distribution")

        save_path = os.path.join(dist_dir, f"{col}_distribution.png")
        plt.savefig(save_path)
        plt.close()

        print(f"Saved: {save_path}")

    print("Distribution plots completed.\n")
