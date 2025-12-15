import os
import seaborn as sns
import matplotlib.pyplot as plt

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def correlation_matrix(df, output_dir):
    # Ensure output directory exists
    ensure_dir(output_dir)

    print("\n=== PLOT: CORRELATION MATRIX ===")

    numeric_df = df.select_dtypes(include=['float64', 'int64'])

    plt.figure(figsize=(14, 10))
    sns.heatmap(numeric_df.corr(), annot=False, cmap="coolwarm")
    plt.title("Correlation Heatmap (Numeric Features Only)")

    save_path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Saved: {save_path}")
    print("Correlation matrix completed.\n")
