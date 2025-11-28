import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def define_preprocessing_and_defaults(X_train):
    """
    Defines feature lists, the ColumnTransformer, and calculates mean defaults.
    Returns: ColumnTransformer instance, MEAN_DEFAULTS dict, FULL_FEATURE_COLUMNS list
    """
    print("\n--- 2. Defining Preprocessing & Imputation Defaults ---")

    # Define numeric & categorical columns
    num_cols = [
        'latitude','longitude','PM2.5','PM10','NO2','SO2','CO','O3',
        'temperature','humidity','wind_speed','wind_dir','dist_to_road',
        'dist_to_industry','dist_to_farm','fire_nearby','fire_count',
        'fire_min_dist_km','Confidence_Score'
    ]
    cat_cols = ['city', 'Season']

    FULL_FEATURE_COLUMNS = num_cols + cat_cols

    # Define Preprocessing Pipeline
    preprocess = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])
    print("Defined ColumnTransformer for scaling and one-hot encoding.")

    # Calculate Mean Defaults for Imputation (excluding lat/lon)
    features_to_impute = [col for col in num_cols if col not in ['latitude', 'longitude']]
    MEAN_DEFAULTS = X_train[features_to_impute].mean().to_dict()
    print("Calculated MEAN_DEFAULTS for imputation from training data.")

    return preprocess, MEAN_DEFAULTS, FULL_FEATURE_COLUMNS