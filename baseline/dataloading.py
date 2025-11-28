import pandas as pd
import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_data_and_split():
    print("\n--- 1. Data Loading and Initial Exploration ---")
    try:
        df = pd.read_csv("data/fixed_dates.csv")
    except FileNotFoundError:
        print("ERROR: File not found. Please ensure the file is in the correct path.")
        sys.exit(1)

    print(df.head())
    print("\nTarget Label Distribution:")
    print(df['Source'].value_counts())

    # Feature and Target Separation
    target = "Source"
    X = df.drop(columns=[target, 'date'])
    y = df[target]

    # Target Encoding
    le = LabelEncoder()
    y = le.fit_transform(y)
    print("Target encoded (0, 1) using LabelEncoder.")

    # Data Splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Data split. Training set size: {len(X_train)}")
    return X_train, X_test, y_train, y_test, le