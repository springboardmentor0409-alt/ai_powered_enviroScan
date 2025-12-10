import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eda.load_data import load_dataset
from eda.missing_values import check_missing
from eda.distributions import plot_distributions
from eda.boxplots import plot_boxplots
from eda.correlation import correlation_matrix
from eda.outliers import detect_outliers
from eda.geospatial import plot_geospatial
from eda.time_trends import plot_time_trends

# --- CONFIG ---
INPUT_PATH = "../data/unlabeled_pollution_data.csv"
OUTPUT_DIR = "../results/eda/"

# Ensure directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- LOAD DATA ---
df = load_dataset(INPUT_PATH, OUTPUT_DIR)

# --- RUN EDA ---
print("Running EDA...")

check_missing(df, output_dir=OUTPUT_DIR)
plot_distributions(df, output_dir=OUTPUT_DIR)
plot_boxplots(df, output_dir=OUTPUT_DIR)
correlation_matrix(df, output_dir=OUTPUT_DIR)
detect_outliers(df, output_dir=OUTPUT_DIR)
plot_geospatial(df, output_dir=OUTPUT_DIR)
plot_time_trends(df, output_dir=OUTPUT_DIR)

print("\n=== EDA Completed Successfully ===")
print(f"All results saved to: {OUTPUT_DIR}")
