import joblib
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

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
    plot_feature_importance,
    plot_cv_scores
)


def train_xgboost(input_path, model_path, results_path, seed=42):

    if XGBClassifier is None:
        raise ImportError("Run: pip install xgboost")

    df = load_dataset(input_path)

    y_enc, le = encode_target(df)
    X, num_cols, cat_cols = get_feature_splits(df)
    preprocessor = build_preprocessor(num_cols, cat_cols)

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", XGBClassifier(
            objective="multi:softprob",
            num_class=len(le.classes_),
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.8,
            gamma=1.0,
            reg_lambda=1.0,
            reg_alpha=0.5,
            eval_metric="mlogloss",
            random_state=seed,
            tree_method="hist"
        ))
    ])

    X_train, X_test, y_train, y_test = prepare_train_test(X, y_enc, seed)

    print("\n Training XGBoost...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    Path(model_path).mkdir(parents=True, exist_ok=True)
    Path(results_path).mkdir(parents=True, exist_ok=True)

    joblib.dump(model, f"{model_path}/xgboost.joblib")
    joblib.dump(le, f"{model_path}/label_encoder.joblib")

    class_names = list(le.classes_)

    # Classification report
    report_txt = classification_report(y_test, y_pred, target_names=class_names)
    print("\n Classification Report:")
    print(report_txt)
    with open(f"{results_path}/classification_report.txt", "w") as f:
        f.write(report_txt)
    pd.DataFrame(classification_report(y_test, y_pred, target_names=class_names, output_dict=True)).to_csv(
        f"{results_path}/classification_report.csv"
    )

    # Confusion matrix + plots
    cm = confusion_matrix(y_test, y_pred)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(f"{results_path}/confusion_matrix.csv")
    plot_confusion_matrix(y_test, y_pred, class_names, f"{results_path}/confusion_matrix.png")
    plot_classification_report(y_test, y_pred, class_names, f"{results_path}/classification_report.png")
    print(f"Saved confusion matrix & classification report plots → {results_path}")

    # Feature importance (XGBoost)
    try:
        xgb = model.named_steps["clf"]
        try:
            feat_names = model.named_steps["preprocessor"].get_feature_names_out()
        except Exception:
            feat_names = list(num_cols) + list(cat_cols)
        plot_feature_importance(xgb, feat_names, f"{results_path}/feature_importance.png", top_n=50)
        print(f"Saved feature importance plot → {results_path}/feature_importance.png")
    except Exception as e:
        print("Warning: could not plot feature importance:", e)

    # Cross-validation
    print("\n Running 5-Fold CV...")
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(model, X, y_enc, cv=kfold, scoring="f1_macro")
    pd.DataFrame({"fold": list(range(1, len(cv_scores)+1)), "f1_macro": cv_scores}).to_csv(
        f"{results_path}/crossval_scores.csv", index=False
    )
    plot_cv_scores(cv_scores, f"{results_path}/cv_f1_scores.png")
    print(f"Saved CV plot → {results_path}/cv_f1_scores.png")

    print("\n XGBoost training & plotting complete\n")


if __name__ == "__main__":
    train_xgboost(
        input_path="../data/labeled_pollution_data.csv",
        model_path="../models/xgboost_model",
        results_path="../results/xgboost_model"
    )
