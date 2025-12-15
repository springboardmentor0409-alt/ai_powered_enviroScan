import os
import pandas as pd
from io import StringIO

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_dataset(path, output_dir):
    ensure_dir(output_dir)

    df = pd.read_csv(path)

    # Fix date parsing (DD-MM-YYYY)
    df['date'] = pd.to_datetime(df['date'], format="%d-%m-%Y", errors='coerce')

    # New date columns
    df['dayofyear'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year

    # -----------------------
    # SAVE HEAD
    # -----------------------
    print("\n=== HEAD ===")
    print(df.head())
    df.head().to_csv(f"{output_dir}/head.csv", index=False)

    # -----------------------
    # SAVE INFO (needs buffer)
    # -----------------------
    print("\n=== INFO ===")
    buffer = StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    print(info_str)

    # save as text file
    with open(f"{output_dir}/info.txt", "w") as f:
        f.write(info_str)

    # -----------------------
    # SAVE SHAPE
    # -----------------------
    print("\n=== SHAPE ===")
    print(df.shape)

    shape_df = pd.DataFrame({"rows": [df.shape[0]], "columns": [df.shape[1]]})
    shape_df.to_csv(f"{output_dir}/shape.csv", index=False)

    # -----------------------
    # SUMMARY STATS
    # -----------------------
    print("\n=== SUMMARY ===")
    summary = df.describe()
    print(summary)
    summary.to_csv(f"{output_dir}/summary.csv")

    return df
