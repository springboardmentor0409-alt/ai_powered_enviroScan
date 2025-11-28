import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import sys
# Imports for handling imbalanced data
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
except ImportError:
    print("FATAL: imbalanced-learn missing. Cannot run SMOTE training.")
    sys.exit(1)


def train_logistic_regression(X_train, X_test, y_train, y_test, preprocess):
    """Trains and evaluates a Logistic Regression model."""
    print("\n>>> Training Logistic Regression...")
    log_model = Pipeline(steps=[
        ('pre', preprocess),
        ('clf', LogisticRegression(max_iter=1000, solver="lbfgs"))
    ])

    log_model.fit(X_train, y_train)
    pred_lr = log_model.predict(X_test)

    print("\n=== Logistic Regression Results ===")
    print(classification_report(y_test, pred_lr))

    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, pred_lr), annot=True, cmap="Blues", fmt='d')
    plt.title("Confusion Matrix - Logistic Regression")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

def train_smote_rf(X_train, X_test, y_train, y_test, preprocess):
    """
    Trains a regularized Random Forest model using SMOTE.
    Returns: The trained ImbPipeline model.
    """
    print("\n>>> Training SMOTE-Balanced Random Forest (Stricter Regularization)...")

    # Stricter, more regularized classifier
    rf_classifier_strict = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',
        max_depth=10,
        min_samples_leaf=10
    )

    # Pipeline with simple SMOTE
    rf_model = ImbPipeline(steps=[
        ('pre', preprocess),
        ('sampler', SMOTE(random_state=42)),
        ('clf', rf_classifier_strict)
    ])

    # Fit the pipeline
    rf_model.fit(X_train, y_train)

    # Evaluate the model
    pred_rf_new = rf_model.predict(X_test)
    print("\n=== Random Forest Results (SMOTE + Regularization) ===")
    print(classification_report(y_test, pred_rf_new))

    # Plotting Confusion Matrix for RF
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, pred_rf_new), annot=True, cmap="Greens", fmt='d')
    plt.title("Confusion Matrix - Random Forest (SMOTE)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    return rf_model

def train_all_models(X_train, X_test, y_train, y_test, preprocess):
    """Runs both model training functions and returns the final chosen model (SMOTE RF)."""
    train_logistic_regression(X_train, X_test, y_train, y_test, preprocess)
    final_model = train_smote_rf(X_train, X_test, y_train, y_test, preprocess)
    return final_model