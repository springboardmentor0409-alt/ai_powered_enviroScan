import matplotlib.pyplot as plt
import seaborn as sns

pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]

def plot_boxplots(df):
    for col in pollutants:
        plt.figure(figsize=(6,4))
        sns.boxplot(x=df[col])
        plt.title(f"{col} Boxplot (Outliers)")
        plt.show()
