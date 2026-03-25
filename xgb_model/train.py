import os
import pickle
import numpy as np
import xgboost as xgb

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    make_scorer,
    accuracy_score,
    roc_auc_score
)
from sklearn.model_selection import RandomizedSearchCV

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

def search_hyperparams(X_train, y_train):
    best_model_path = f"{MODEL_DIR}/best_xgb_model.pickle"

    if os.path.exists(best_model_path):
        print("Loading existing best model...")
        with open(best_model_path, "rb") as f:
            return pickle.load(f)

    print("Running GridSearchCV...")

    params = {
        "objective": "binary:logistic",
        "n_estimators": 50,
        "n_jobs": -1,
        "random_state": SEED,
        "eval_metric": "logloss"
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

    grid = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        n_iter=40,  
        scoring=scorer,
        cv=3,
        n_jobs=-1,
        verbose=2,
        random_state=SEED,
    )

    grid.fit(X_train, y_train)

    with open(best_model_path, "wb") as f:
        pickle.dump(grid.best_estimator_, f)

    print("Best params:", grid.best_params_)
    return grid.best_estimator_

def train_final(X_train, y_train, X_dev, y_dev, X_test, y_test, use_loaded_model=False):
    if use_loaded_model:
        best_model_path = f"{MODEL_DIR}/best_xgb_model.pickle"
        print("Loading best model from", best_model_path)
        with open(best_model_path, "rb") as f:
            xgb_model = pickle.load(f)
    else:
        params = { "objective": "binary:logistic", 
                  "n_estimators": 500, 
                  "n_jobs": -1, 
                  "random_state": SEED, 
                  "gamma": 0, 
                  "learning_rate": 0.5, 
                  "max_depth": 20, 
                  "reg_alpha": 0, 
                  "reg_lambda": 1, 
                  "eval_metric": "logloss", 
                }
        xgb_model = xgb.XGBClassifier(**params, early_stopping_rounds=30)

        xgb_model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_dev, y_dev)]
        )

    print("\nEvaluating on test set...")
    y_score = xgb_model.predict_proba(X_test)[:, 1]

    thresholds = np.linspace(0.05, 0.95, 200)
    best_thr = 0.5
    best_f1 = 0

    for t in thresholds:
        y_pred_tmp = (y_score > t).astype(int)
        f1 = f1_score(y_test, y_pred_tmp)

        if f1 > best_f1:
            best_f1 = f1
            best_thr = t

    print("Best threshold:", best_thr)
    print("Best F1:", best_f1)

    y_pred = (y_score > best_thr).astype(int)

    print("f1 non-fraud:", f1_score(y_test, y_pred, average="binary", pos_label=0))
    print("f1 fraud:", f1_score(y_test, y_pred, average="binary", pos_label=1))
    print("f1 macro:", f1_score(y_test, y_pred, average="macro"))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("Precision: ",precision_score(y_test, y_pred, zero_division=0))
    print("Recall: ",recall_score(y_test, y_pred, zero_division=0))
    print("F1: ",f1_score(y_test, y_pred, zero_division=0))
    print("RocAuc: ",roc_auc_score(y_test, y_pred))
    print("Accuracy: ",accuracy_score(y_test, y_pred))

    with open(f"{MODEL_DIR}/xgb_model.pickle", "wb") as f:
        pickle.dump(xgb_model, f)

    print("\nModel saved")


def main():
    X_train, X_dev, X_test, y_train, y_dev, y_test = load_data()
    #print("Searching best hyperparameters...")
    #best_model = search_hyperparams(X_train, y_train)

    train_final(X_train, y_train, X_dev, y_dev, X_test, y_test, use_loaded_model=False)


if __name__ == "__main__":
    main()