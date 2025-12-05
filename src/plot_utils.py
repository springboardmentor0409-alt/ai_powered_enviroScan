import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report


# CONFUSION MATRIX HEATMAP
def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", linewidths=1)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# CLASSIFICATION REPORT HEATMAP
def plot_classification_report(y_true, y_pred, class_names, save_path):
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )

    df = pd.DataFrame(report_dict).iloc[:-1, :]  # remove accuracy row

    plt.figure(figsize=(10, 6))
    sns.heatmap(df.iloc[:, :-1], annot=True, cmap="Greens")  # drop support column for heatmap
    plt.title("Classification Report (Precision / Recall / F1)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# FEATURE IMPORTANCE PLOT (RF + XGB)
def plot_feature_importance(model, feature_names, save_path, top_n=20):
    if not hasattr(model, "feature_importances_"):
        print("Model does not provide feature_importances_. Skipping plot.")
        return

    importances = model.feature_importances_
    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 8))
    sns.barplot(x="importance", y="feature", data=fi)
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# CROSS-VALIDATION SCORES PLOT
def plot_cv_scores(cv_scores, save_path):
    plt.figure(figsize=(8, 5))
    sns.lineplot(x=np.arange(1, len(cv_scores) + 1), y=cv_scores, marker="o")
    plt.title("Cross-Validation F1-Macro Scores")
    plt.xlabel("Fold Number")
    plt.ylabel("F1 Macro Score")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
