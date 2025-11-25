import seaborn as sns
import matplotlib.pyplot as plt

def correlation_matrix(df):
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    plt.figure(figsize=(14, 10))
    sns.heatmap(numeric_df.corr(), annot=False, cmap="coolwarm")
    plt.title("Correlation Heatmap (Numeric Features Only)")
    plt.show()
