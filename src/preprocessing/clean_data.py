import pandas as pd

def load_raw_data(path):
    """Load raw pollution dataset."""
    df = pd.read_csv(path)
    return df


def clean_pollution_data(df):
    """Clean missing values, fix date formats, remove duplicates, validate coordinates."""

    # Fix date column
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Remove rows with invalid coordinates
    df = df[(df['latitude'].between(-90, 90)) & 
            (df['longitude'].between(-180, 180))]

    # Numeric column cleanup
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_clean_data(df, path):
    df.to_csv(path, index=False)
    print(f"[✔] Cleaned dataset saved to: {path}")
