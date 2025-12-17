import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eda.load_data import load_dataset
from eda.missing_values import check_missing
from eda.summary_stats import summary_stats
from eda.distributions import plot_distributions
from eda.boxplots import plot_boxplots
from eda.correlation import correlation_matrix
from eda.outliers import detect_outliers
from eda.geospatial import plot_geospatial
from eda.time_trends import plot_time_trends

# RUN EDA
df = load_dataset("data/EnviroFinal_final_unlabled.csv")

check_missing(df)
summary_stats(df)
plot_distributions(df)
plot_boxplots(df)
correlation_matrix(df)
detect_outliers(df)
plot_geospatial(df)
plot_time_trends(df)

print("\n=== EDA Completed Successfully ===")
