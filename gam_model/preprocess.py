import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42
TOP_K_COUNTRIES = 20
MODEL_DIR = './gam_model'


def load_and_merge(transactions_path, accounts_path):
    tx = pd.read_csv(transactions_path)
    acc = pd.read_csv(accounts_path)

    df = tx.merge(
        acc,
        left_on="SENDER_ACCOUNT_ID",
        right_on="ACCOUNT_ID",
        how="left",
        suffixes=("", "_ACC")
    )

    return df


def add_time_features(df):
    seconds_in_day = 24 * 60 * 60
    time_in_day = df["TIMESTAMP"] % seconds_in_day

    df["TX_TIME_SIN"] = np.sin(2 * np.pi * time_in_day / seconds_in_day)
    df["TX_TIME_COS"] = np.cos(2 * np.pi * time_in_day / seconds_in_day)

    return df


def preprocess_numeric(df):
    df["TX_AMOUNT"] = np.log1p(df["TX_AMOUNT"])
    df["INIT_BALANCE"] = np.log1p(df["INIT_BALANCE"])
    return df


def preprocess_categorical(df):
    top_countries = (
        df["COUNTRY"]
        .value_counts()
        .nlargest(TOP_K_COUNTRIES)
        .index
    )
    df["COUNTRY"] = df["COUNTRY"].where(
        df["COUNTRY"].isin(top_countries),
        other="OTHER"
    )
    return df


def build_feature_matrix(df, encoder=None, fit_encoder=False):
    num_features = [
        "TX_AMOUNT",
        "INIT_BALANCE",
        "TX_TIME_SIN",
        "TX_TIME_COS",
    ]

    cat_features = [
        "TX_TYPE",
        "ACCOUNT_TYPE",
        "COUNTRY",
        "TX_BEHAVIOR_ID"
    ]

    X_num = df[num_features].values

    if fit_encoder:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )
        X_cat = encoder.fit_transform(df[cat_features])
    else:
        X_cat = encoder.transform(df[cat_features])

    X = np.hstack([X_num, X_cat])

    feature_names = (
        num_features +
        list(encoder.get_feature_names_out(cat_features))
    )

    return X, encoder, feature_names


def main():
    df = load_and_merge(
        "./dataset/transactions.csv",
        "./dataset/accounts.csv"
    )

    df = preprocess_numeric(df)
    df = preprocess_categorical(df)
    df = add_time_features(df)

    target = df["IS_FRAUD"].astype(int)
    
    print(df.head(5))

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=target,
        random_state=RANDOM_STATE
    )

    X_train, encoder, feature_names = build_feature_matrix(
        train_df,
        fit_encoder=True
    )
    X_val, _, _ = build_feature_matrix(
        val_df,
        encoder=encoder,
        fit_encoder=False
    )

    y_train = train_df["IS_FRAUD"].values
    y_val = val_df["IS_FRAUD"].values

    np.save(f"{MODEL_DIR}/dataframes/X_train.npy", X_train)
    np.save(f"{MODEL_DIR}/dataframes/X_val.npy", X_val)
    np.save(f"{MODEL_DIR}/dataframes/y_train.npy", y_train)
    np.save(f"{MODEL_DIR}/dataframes/y_val.npy", y_val)

    joblib.dump(encoder, f"{MODEL_DIR}/joblibs/encoder.joblib")
    joblib.dump(feature_names, f"{MODEL_DIR}/joblibs/feature_names.joblib")

    print("Preprocessing finished")
    print(f"X_train shape: {X_train.shape}")
    print(f"Fraud rate (val): {y_val.mean():.4f}")


if __name__ == "__main__":
    main()
