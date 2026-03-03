import os
import pickle
import numpy as np
import xgboost as xgb
from imblearn.over_sampling import SMOTE

from sklearn.metrics import f1_score, make_scorer, confusion_matrix
from sklearn.model_selection import GridSearchCV

MODEL_DIR = "./xgb_model"
SEED = 1212

def load_data():
    X_train = np.load(f"{MODEL_DIR}/dataframes/X_train.npy")
    X_dev = np.load(f"{MODEL_DIR}/dataframes/X_dev.npy")
    X_test = np.load(f"{MODEL_DIR}/dataframes/X_test.npy")

    y_train = np.load(f"{MODEL_DIR}/dataframes/y_train.npy")
    y_dev = np.load(f"{MODEL_DIR}/dataframes/y_dev.npy")
    y_test = np.load(f"{MODEL_DIR}/dataframes/y_test.npy")

    return X_train, X_dev, X_test, y_train, y_dev, y_test

def f1_score_custom(y_pred, y_true):
    labels = None
    preds = None

    if hasattr(y_true, "get_label"):
        labels = y_true.get_label()
        preds = y_pred
    elif hasattr(y_pred, "get_label"):
        labels = y_pred.get_label()
        preds = y_true
    else:
        def is_binary_array(arr):
            arr_u = np.unique(arr)
            return np.all(np.isin(arr_u, [0, 1]))

        a = np.asarray(y_pred).ravel()
        b = np.asarray(y_true).ravel()

        if is_binary_array(a) and not is_binary_array(b):
            labels = a
            preds = b
        else:
            labels = b
            preds = a

    preds = np.asarray(preds).ravel()
    labels = np.asarray(labels).ravel()

    if np.issubdtype(preds.dtype, np.floating) or (preds.max() > 1):
        preds_bin = (preds > 0.5).astype(int)
    else:
        preds_bin = (preds > 0.5).astype(int)

    try:
        f1 = f1_score(labels, preds_bin, average="binary")
    except ValueError:
        labels_bin = (labels > 0.5).astype(int)
        f1 = f1_score(labels_bin, preds_bin, average="binary")

    return "f1_err", 1.0 - f1

def oversample_train(X_train, y_train):
    print("\nBefore oversampling:")
    unique, counts = np.unique(y_train, return_counts=True)
    print(dict(zip(unique, counts)))

    smote = SMOTE(
        sampling_strategy=0.5, 
        random_state=SEED,
        k_neighbors=5
    )

    X_res, y_res = smote.fit_resample(X_train, y_train)

    print("\nAfter oversampling:")
    unique, counts = np.unique(y_res, return_counts=True)
    print(dict(zip(unique, counts)))

    return X_res, y_res

def search_hyperparams(X_train, y_train):
    best_model_path = f"{MODEL_DIR}/best_xgb_model.pickle"

    if os.path.exists(best_model_path):
        print("Loading existing best model...")
        with open(best_model_path, "rb") as f:
            return pickle.load(f)

    print("Running GridSearchCV...")

    params = {
        "objective": "binary:logistic",
        "n_estimators": 100,
        "n_jobs": -1,
        "random_state": SEED,
    }

    model = xgb.XGBClassifier(**params)

    param_grid = {
        "learning_rate": [0.01, 0.1, 0.5, 1],
        "max_depth": [5, 10, 15, 20],
        "gamma": [0, 0.1, 0.5, 1],
        "reg_alpha": [0, 0.1, 0.5, 1],
        "reg_lambda": [0, 0.1, 0.5, 1],
    }

    scorer = make_scorer(f1_score, greater_is_better=True)

    grid = GridSearchCV(
        model,
        param_grid,
        scoring=scorer,
        cv=3,
        n_jobs=-1,
        verbose=2,
    )

    grid.fit(X_train, y_train)

    with open(best_model_path, "wb") as f:
        pickle.dump(grid.best_estimator_, f)

    print("Best params:", grid.best_params_)
    return grid.best_estimator_

def train_final(model_or_none, X_train, y_train, X_dev, y_dev, X_test, y_test, use_loaded_model=False):
    if use_loaded_model:
        best_model_path = f"{MODEL_DIR}/best_xgb_model.pickle"
        print("Loading best model from", best_model_path)
        with open(best_model_path, "rb") as f:
            xgb_model = pickle.load(f)
    else:
        params = {
            "objective": "binary:logistic",
            "n_estimators": 200,
            "n_jobs": -1,
            "random_state": SEED,
            "gamma": 0,
            "learning_rate": 0.5,
            "max_depth": 10,
            "reg_alpha": 0.5,
            "reg_lambda": 0,
            "eval_metric": f1_score_custom
        }
        xgb_model = xgb.XGBClassifier(**params)

        xgb_model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_dev, y_dev)]
        )

    print("\nEvaluating on test set...")
    y_pred = xgb_model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)

    print("f1 non-fraud:", f1_score(y_test, y_pred, average="binary", pos_label=0))
    print("f1 fraud:", f1_score(y_test, y_pred, average="binary", pos_label=1))
    print("f1 macro:", f1_score(y_test, y_pred, average="macro"))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    with open(f"{MODEL_DIR}/xgb_model.pickle", "wb") as f:
        pickle.dump(xgb_model, f)

    print("\nModel saved")


def main():
    X_train, X_dev, X_test, y_train, y_dev, y_test = load_data()
    X_train, y_train = oversample_train(X_train, y_train)
    print("Searching best hyperparameters...")
    best_model = search_hyperparams(X_train, y_train)

    train_final(best_model, X_train, y_train, X_dev, y_dev, X_test, y_test, use_loaded_model=True)


if __name__ == "__main__":
    main()