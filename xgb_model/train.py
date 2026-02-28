import numpy as np
import joblib
import xgboost as xgb

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    auc,
)
from sklearn.model_selection import train_test_split

MODEL_DIR = "./xgb_model"

def load_data():
    X = np.load(f"{MODEL_DIR}/dataframes/X.npy")
    y = np.load(f"{MODEL_DIR}/dataframes/y.npy")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return X_train, X_val, y_train, y_val

def select_threshold(y_true, y_prob, max_fp_allowed=20):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    fraud_count = y_true.sum()

    best_threshold = 0.5
    best_recall = 0
    best_precision = 0
    best_fp = 0

    for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
        tp = r * fraud_count
        fp = tp * (1 - p) / p if p > 0 else 0

        if fp <= max_fp_allowed and r > best_recall:
            best_recall = r
            best_threshold = t
            best_precision = p
            best_fp = fp

    print(f"Selected threshold: {best_threshold:.4f} (recall={best_recall:.4f}, precision={best_precision:.4f}, FP={int(best_fp)})")
    return best_threshold

def print_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()
    total = cm.sum()
    actual_pos = TP + FN

    print("\n==== Confusion Matrix ====")
    print(f"TP = {TP}, FP = {FP}, FN = {FN}, TN = {TN}")
    print(f"Total = {total}")
    print(f"Precision = {precision_score(y_true, y_pred):.6f}")
    print(f"Recall    = {recall_score(y_true, y_pred):.6f}")
    print(f"F1        = {f1_score(y_true, y_pred):.6f}")
    print(f"Prevalence (pos/total) = {actual_pos / total:.6f}")
    print("\nMatrix (counts and % of total):")
    print(f"TN = {TN} ({TN/total*100:.2f}%), FP = {FP} ({FP/total*100:.2f}%)")
    print(f"FN = {FN} ({FN/total*100:.2f}%), TP = {TP} ({TP/total*100:.5f}%)")
    return cm

def main():
    X_train, X_val, y_train, y_val = load_data()

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Scale pos weight: {scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='aucpr' 
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)]
    )

    y_val_prob = model.predict_proba(X_val)[:, 1]

    threshold = select_threshold(y_val, y_val_prob)
    y_pred = (y_val_prob >= threshold).astype(int)

    print_confusion_matrix(y_val, y_pred)

    precisions, recalls, _ = precision_recall_curve(y_val, y_val_prob)
    pr_auc = auc(recalls, precisions)
    print(f"\nPrecision-Recall AUC: {pr_auc:.6f}")

    joblib.dump(model, f"{MODEL_DIR}/joblibs/xgb_model.joblib")
    joblib.dump(threshold, f"{MODEL_DIR}/joblibs/decision_threshold.joblib")
    print("\nXGBoost model and threshold saved!")

if __name__ == "__main__":
    main()