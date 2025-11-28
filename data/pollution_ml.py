
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("labeled_pollution_data.csv")

print(df.head())
print(df.info())
print(df.describe())
print(df['Source'].value_counts())   # label distribution

print(df.columns)

# Date preprocessing

# drop duplicates
df = df.drop_duplicates()

# handle missing values
df = df.fillna(df.median(numeric_only=True))


# convert date to datetime and create numeric time features
df['date'] = pd.to_datetime(df['date'])

df['year']      = df['date'].dt.year
df['month']     = df['date'].dt.month
df['dayofyear'] = df['date'].dt.dayofyear


# Separate features & target
target = "Source"
X = df.drop(columns=[target, 'date'])  
y = df[target]

# label encode target
le = LabelEncoder()
y = le.fit_transform(y)


# Define numeric & categorical columns

num_cols = [
    'latitude','longitude','PM2.5','PM10','NO2','SO2','CO','O3',
    'temperature','humidity','wind_speed','wind_dir','dist_to_road',
    'dist_to_industry','dist_to_farm','fire_nearby','fire_count',
    'fire_min_dist_km','Confidence_Score',
    'year','month','dayofyear'      
]

cat_cols = ['city', 'Season']        

print("Numeric columns:", num_cols)
print("Categorical columns:", cat_cols)
preprocess = ColumnTransformer(transformers=[
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Logistic Regression Baseline

print("\n>>> Training Logistic Regression...")
log_model = Pipeline(steps=[
    ('pre', preprocess),
    ('clf', LogisticRegression(max_iter=1000, solver="lbfgs"))
])

log_model.fit(X_train, y_train)
pred_lr = log_model.predict(X_test)

print("\n Logistic Regression Results")
print(classification_report(y_test, pred_lr))

plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, pred_lr), annot=True, cmap="Blues", fmt='d')
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# 5) Random Forest Baseline

print("\n>>> Training Random Forest...")
rf_model = Pipeline(steps=[
    ('pre', preprocess),
    ('clf', RandomForestClassifier(
        n_estimators=200,         
        random_state=42,
        n_jobs=-1                
    ))
])

rf_model.fit(X_train, y_train)
pred_rf = rf_model.predict(X_test)

print("\n Random Forest Results")
print(classification_report(y_test, pred_rf))

plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, pred_rf), annot=True, cmap="Greens", fmt='d')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()