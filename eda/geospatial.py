import os
import matplotlib.pyplot as plt

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def plot_geospatial(df, output_dir):
    # Ensure output directory exists
    ensure_dir(output_dir)

    print("\n=== PLOT: GEOSPATIAL SCATTER ===")

    plt.figure(figsize=(8, 6))
    plt.scatter(df["longitude"], df["latitude"], s=2, alpha=0.5)
    plt.title("Geospatial Scatter — Data Points")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    save_path = os.path.join(output_dir, "geospatial_scatter.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Saved: {save_path}")
    print("Geospatial plot completed.\n")
