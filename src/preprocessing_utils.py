import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


# Load dataset
def load_dataset(input_path):
    df = pd.read_csv(input_path)
    return df


# Target encoding (LabelEncoder)
def encode_target(df):
    y = df["source_label"]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return y_enc, le


# Split features into numerical & categorical
def get_feature_splits(df):

    drop_cols = ["source_label", "date", "latitude", "longitude"]
    X = df.drop(columns=drop_cols, errors="ignore")

    cat_cols = ["city", "location_id", "season"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    return X, num_cols, cat_cols

# Build ColumnTransformer (shared across all models)
def build_preprocessor(num_cols, cat_cols):

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    return preprocessor

# Train-test split helper
def prepare_train_test(X, y_enc, seed=42):
    return train_test_split(
        X, y_enc, 
        test_size=0.2, 
        random_state=seed, 
        stratify=y_enc
    )
