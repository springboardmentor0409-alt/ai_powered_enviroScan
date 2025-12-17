import joblib
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold

from preprocessing_utils import (
    load_dataset,
    encode_target,
    get_feature_splits,
    build_preprocessor,
    prepare_train_test
)

from plot_utils import (
    plot_confusion_matrix,
    plot_classification_report,
    plot_cv_scores
)


def train_logistic_regression(input_path, model_path, results_path, seed=42):

    print("\n Loading dataset...")
    df = load_dataset(input_path)

    # Preprocessing
    y_enc, le = encode_target(df)
    X, num_cols, cat_cols = get_feature_splits(df)
    preprocessor = build_preprocessor(num_cols, cat_cols)

    # Model pipeline
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(max_iter=2000, n_jobs=-1, class_weight="balanced"))
    ])

    # Train/Test split
    X_train, X_test, y_train, y_test = prepare_train_test(X, y_enc, seed)

    print("\n Training Logistic Regression...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Output directories
    Path(model_path).mkdir(parents=True, exist_ok=True)
    Path(results_path).mkdir(parents=True, exist_ok=True)

    # Save model + encoder
    joblib.dump(model, f"{model_path}/logistic_regression.joblib")
    joblib.dump(le, f"{model_path}/label_encoder.joblib")

    class_names = list(le.classes_)

    # Classification report 
    report_txt = classification_report(y_test, y_pred, target_names=class_names)
    print("\nClassification Report:")
    print(report_txt)
    with open(f"{results_path}/classification_report.txt", "w") as f:
        f.write(report_txt)
    pd.DataFrame(classification_report(y_test, y_pred, target_names=class_names, output_dict=True)).to_csv(
        f"{results_path}/classification_report.csv"
    )

    # Confusion matrix
    cm = classification_matrix = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(f"{results_path}/confusion_matrix.csv")
    print("\n Confusion Matrix saved to CSV.")
    plot_confusion_matrix(y_test, y_pred, class_names, f"{results_path}/confusion_matrix.png")
    print(f"Saved confusion matrix plot → {results_path}/confusion_matrix.png")

    # Classification report heatmap
    plot_classification_report(y_test, y_pred, class_names, f"{results_path}/classification_report.png")
    print(f"Saved classification report heatmap → {results_path}/classification_report.png")

    # Cross-validation
    print("\n Running 5-Fold CV...")
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(model, X, y_enc, cv=kfold, scoring="f1_macro")
    pd.DataFrame({"fold": list(range(1, len(cv_scores)+1)), "f1_macro": cv_scores}).to_csv(
        f"{results_path}/crossval_scores.csv", index=False
    )
    print("CV scores saved →", f"{results_path}/crossval_scores.csv")

    # CV plot
    plot_cv_scores(cv_scores, f"{results_path}/cv_f1_scores.png")
    print(f"Saved CV F1 scores plot → {results_path}/cv_f1_scores.png")

    print("\n Logistic Regression training & plotting complete!\n")


if __name__ == "__main__":
    train_logistic_regression(
        input_path="../data/labeled_pollution_data.csv",
        model_path="../models/logistic_regression",
        results_path="../results/logistic_regression"
    )
