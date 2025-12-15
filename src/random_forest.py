import joblib
import pandas as pd
from pathlib import Path

# Prevent Tkinter / backend thread errors
import matplotlib
matplotlib.use("Agg")

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectFromModel

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


def train_random_forest(input_path, model_path, results_path, seed=42):

    # ---------------- LOAD & PREPARE DATA ----------------
    df = load_dataset(input_path)

    y_enc, le = encode_target(df)
    X, num_cols, cat_cols = get_feature_splits(df)
    preprocessor = build_preprocessor(num_cols, cat_cols)

    # ---------------- IMPROVED RANDOM FOREST --------------
    clf = RandomForestClassifier(
        n_estimators=800,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        oob_score=True,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("feature_select", SelectFromModel(
            RandomForestClassifier(n_estimators=300, random_state=seed)
        )),
        ("clf", clf)
    ])
    # ------------------------------------------------------

    X_train, X_test, y_train, y_test = prepare_train_test(X, y_enc, seed)

    print("\n🔹 Training Random Forest with optimized parameters...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ---------------- SAFE OOB SCORE PRINT ----------------
    try:
        print("\n📌 OOB Score (approx accuracy):", model.named_steps["clf"].oob_score_)
    except Exception:
        print("\n⚠ OOB score unavailable (maybe classifier not fitted or OOB disabled).")

    # ---------------- CREATE DIRECTORIES ------------------
    Path(model_path).mkdir(parents=True, exist_ok=True)
    Path(results_path).mkdir(parents=True, exist_ok=True)

    # ---------------- SAVE MODEL --------------------------
    joblib.dump(model, f"{model_path}/random_forest.joblib")
    joblib.dump(le, f"{model_path}/label_encoder.joblib")
    print(f"\n💾 Model saved → {model_path}/random_forest.joblib")
    print(f"💾 Label encoder saved → {model_path}/label_encoder.joblib")

    # ---------------- CLASSIFICATION REPORT ----------------
    class_names = list(le.classes_)
    report_txt = classification_report(
        y_test, y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names
    )
    print("\n📌 Classification Report:")
    print(report_txt)

    with open(f"{results_path}/classification_report.txt", "w") as f:
        f.write(report_txt)

    pd.DataFrame(
        classification_report(
            y_test, y_pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
            output_dict=True
        )
    ).to_csv(f"{results_path}/classification_report.csv")

    # ---------------- CONFUSION MATRIX ----------------
    cm = confusion_matrix(y_test, y_pred)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        f"{results_path}/confusion_matrix.csv"
    )
    plot_confusion_matrix(y_test, y_pred, class_names, f"{results_path}/confusion_matrix.png")
    plot_classification_report(y_test, y_pred, class_names, f"{results_path}/classification_report.png")
    print(f"📊 Saved confusion matrix & classification report images → {results_path}")

    # ---------------- FEATURE IMPORTANCE (SAFE) ----------------
    try:
        rf = model.named_steps["clf"]
        try:
            all_feature_names = model.named_steps["preprocessor"].get_feature_names_out()
        except:
            all_feature_names = [f"f{i}" for i in range(rf.n_features_)]
        selector_mask = model.named_steps["feature_select"].get_support()
        selected_feature_names = all_feature_names[selector_mask]

        plot_feature_importance(
            rf,
            selected_feature_names,
            f"{results_path}/feature_importance.png",
            top_n=len(selected_feature_names)
        )
        print(f"📌 Saved feature importance plot → {results_path}/feature_importance.png")
    except Exception as e:
        print("⚠ Feature importance plot skipped:", e)

    # ---------------- CROSS VALIDATION ----------------
    print("\n⏳ Running 3-Fold CV (weighted F1)...")
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(
        model, X, y_enc,
        cv=kfold,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=2
    )

    pd.DataFrame({"fold": list(range(1, len(cv_scores)+1)), "f1_weighted": cv_scores}).to_csv(
        f"{results_path}/crossval_scores.csv", index=False
    )
    plot_cv_scores(cv_scores, f"{results_path}/cv_f1_scores.png")
    print(f"📌 Saved CV plot → {results_path}/cv_f1_scores.png")

    print("\n🎉 Random Forest training complete! All model files and plots saved.\n")


if __name__ == "__main__":
    train_random_forest(
        input_path="../data/labeled_pollution_data.csv",
        model_path="../models/random_forest",
        results_path="../results/random_forest"
    )
