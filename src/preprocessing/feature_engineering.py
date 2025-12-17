import pandas as pd

def apply_time_features(df):
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    return df


def apply_ratios(df):
    df['pm_ratio'] = df['PM2.5'] / (df['PM10'] + 1)
    return df


def apply_feature_engineering(df):
    df = apply_time_features(df)
    df = apply_ratios(df)
    return df
