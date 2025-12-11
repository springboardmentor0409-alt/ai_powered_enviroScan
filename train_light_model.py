import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

print("📥 Loading dataset...")
df = pd.read_csv("data/labeled_pollution_data.csv")

# Check column names
required_cols = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "traffic_index"]
missing = [c for c in required_cols if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

print("✅ Columns verified.")

# Select features
X = df[required_cols]

# Detect correct label column
if "source_label" in df.columns:
    y = df["source_label"]
elif "Source" in df.columns:
    y = df["Source"]
else:
    raise ValueError("No label column found. Expected 'source_label' or 'Source'.")

# Encode label
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

print("🚀 Training lightweight Random Forest model...")
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    max_depth=12
)
model.fit(X_train, y_train)

# Ensure /models folder exists
if not os.path.exists("models"):
    os.makedirs("models")

# Save model and encoder
joblib.dump(model, "models/light_model.joblib")
joblib.dump(label_encoder, "models/light_label_encoder.joblib")

print("\n🎉 TRAINING COMPLETE!")
print("📦 Saved: models/light_model.joblib")
print("📦 Saved: models/light_label_encoder.joblib")
