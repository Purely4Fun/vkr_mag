import numpy as np
import joblib
from pygam import LogisticGAM, s, f, l
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve
)
POS_WEIGHT = 500         
TARGET_FP_COST = 7
MAX_TRAIN_SAMPLES = 500000
RANDOM_STATE = 42
MODEL_DIR = './gam_model'

def load_data():
    X_train = np.load(f"{MODEL_DIR}/dataframes/X_train.npy")
    X_val = np.load(f"{MODEL_DIR}/dataframes/X_val.npy")
    y_train = np.load(f"{MODEL_DIR}/dataframes/y_train.npy")
    y_val = np.load(f"{MODEL_DIR}/dataframes/y_val.npy")
    feature_names = joblib.load(f"{MODEL_DIR}/joblibs/feature_names.joblib")

    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)

    encoder = joblib.load(f"{MODEL_DIR}/joblibs/encoder.joblib")
    return X_train, X_val, y_train, y_val, feature_names, encoder


def subsample(X, y, max_samples=MAX_TRAIN_SAMPLES):
    if len(y) <= max_samples:
        return X, y

    rng = np.random.RandomState(RANDOM_STATE)

    fraud_idx = np.where(y == 1)[0]
    legit_idx = np.where(y == 0)[0]

    n_legit = max_samples - len(fraud_idx)
    n_legit = max(n_legit, 0)

    keep_legit = rng.choice(
        legit_idx,
        size=n_legit,
        replace=False
    )

    idx = np.concatenate([fraud_idx, keep_legit])
    rng.shuffle(idx)

    return X[idx], y[idx]

def build_gam(feature_names):
    terms = (
        s(0, n_splines=6) + 
        s(1, n_splines=6) +
        l(2) +
        l(3)
    )

    for i in range(4, len(feature_names)):
        terms += l(i)

    return LogisticGAM(
        terms,
        max_iter=500,
        verbose=True
    )

def compute_sample_weights(y):
    weights = np.ones_like(y, dtype=np.float32)
    weights[y == 1] = POS_WEIGHT
    return weights

def optimize_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    best_cost = float("inf")
    best_threshold = 0.5

    fraud_count = y_true.sum()

    for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
        tp = r * fraud_count
        fp = tp * (1 - p) / p if p > 0 else 0
        fn = fraud_count - tp

        cost = fp * TARGET_FP_COST + fn

        if cost < best_cost:
            best_cost = cost
            best_threshold = t

    return best_threshold

def evaluate(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return precision, recall, f1, cm

def main():
    X_train, X_val, y_train, y_val, feature_names, encoder = load_data()

    print(f"Original train shape: {X_train.shape}")
    print(f"Validation shape:     {X_val.shape}")
    print(f"Fraud rate (val):     {y_val.mean():.4f}")

    X_train, y_train = subsample(X_train, y_train)
    print(f"Subsampled train:     {X_train.shape}")

    sample_weights = compute_sample_weights(y_train)

    gam = build_gam(feature_names)

    print("Training Logistic GAM...")
    gam.fit(X_train, y_train, weights=sample_weights)

    print("Training finished")

    y_val_prob = gam.predict_proba(X_val)

    threshold = optimize_threshold(y_val, y_val_prob)
    print(f"Optimal threshold: {threshold:.6f}")

    precision, recall, f1, cm = evaluate(y_val, y_val_prob, threshold)

    print("==== GAM Metrics ====")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("Confusion matrix:")
    print(cm)
    print(f"TP: {cm[1,1]}, FP: {cm[0,1]}, FN: {cm[1,0]}, TN: {cm[0,0]}")

    CAT_FEATURES = [
        "TX_TYPE",
        "ACCOUNT_TYPE",
        "COUNTRY",
        "TX_BEHAVIOR_ID",
    ]

    cat_mapping = {}

    for i, feature in enumerate(CAT_FEATURES):
        categories = encoder.categories_[i]
        cat_mapping[feature] = {
            cat: idx for idx, cat in enumerate(categories)
        }

    joblib.dump(
        cat_mapping,
        f"{MODEL_DIR}/joblibs/cat_mapping.joblib"
    )

    joblib.dump(gam, f"{MODEL_DIR}/joblibs/gam_model.joblib")
    joblib.dump(threshold, f"{MODEL_DIR}/joblibs/decision_threshold.joblib")

    print("Model saved successfully")

if __name__ == "__main__":
    main()
