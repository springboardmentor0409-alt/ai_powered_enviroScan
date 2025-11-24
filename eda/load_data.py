import pandas as pd

def load_dataset(path):
    df = pd.read_csv(path)

    # FIX: properly parse DD-MM-YYYY format
    df['date'] = pd.to_datetime(df['date'], format="%d-%m-%Y", errors='coerce')

    # Create new datetime-based columns
    df['dayofyear'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year

    print("\n=== HEAD ===")
    print(df.head())

    print("\n=== INFO ===")
    print(df.info())

    print("\n=== SHAPE ===")
    print(df.shape)

    return df
