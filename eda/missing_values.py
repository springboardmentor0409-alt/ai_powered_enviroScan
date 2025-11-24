import seaborn as sns
import matplotlib.pyplot as plt

def check_missing(df):
    print("\n=== MISSING VALUES ===")
    print(df.isnull().sum())

    sns.heatmap(df.isnull(), cbar=False)
    plt.title("Missing Values Heatmap")
    plt.show()
