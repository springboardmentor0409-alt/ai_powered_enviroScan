import matplotlib.pyplot as plt
import seaborn as sns

pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]

def plot_distributions(df):
    for col in pollutants:
        plt.figure(figsize=(7, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f"{col} Distribution")
        plt.show()
