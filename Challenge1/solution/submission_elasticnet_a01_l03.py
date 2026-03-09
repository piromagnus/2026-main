import argparse
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

GENDER_COLUMN = "gender"
MODEL_CONFIG = {
    "alpha": 0.01,
    "l1_ratio": 0.3,
    "max_iter": 10000,
    "tol": 1e-3,
    "selection": "random",
    "random_state": 42,
}


def get_train_data(input_dir):
    x_train = pd.read_csv(os.path.join(input_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(input_dir, "y_train.csv")).squeeze("columns")
    return x_train, y_train


def get_test_data(input_dir):
    return pd.read_csv(os.path.join(input_dir, "X_test.csv"))


def build_preprocess(x_frame):
    numeric_columns = [column for column in x_frame.columns if column != GENDER_COLUMN]
    categorical_columns = [GENDER_COLUMN] if GENDER_COLUMN in x_frame.columns else []

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "one_hot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )


def build_model(x_frame):
    preprocess = build_preprocess(x_frame)
    regressor = ElasticNet(**MODEL_CONFIG)
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("regressor", regressor),
        ]
    )


def save_predictions(predictions, output_dir, expected_rows):
    if len(predictions) != expected_rows:
        raise ValueError(
            "Prediction length mismatch: "
            f"expected {expected_rows}, got {len(predictions)}"
        )

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "y_pred.csv")
    pd.DataFrame({"age": predictions}).to_csv(output_path, index=False)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Submission-only script: train ElasticNet a01-l03 on full train data."
    )
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    args = parser.parse_args()

    print("Reading data")
    x_train, y_train = get_train_data(args.input_dir)
    x_test = get_test_data(args.input_dir)

    print("Training fixed model: ElasticNet(alpha=0.01, l1_ratio=0.3)")
    model = build_model(x_train)
    model.fit(x_train, np.asarray(y_train).ravel())

    print("Predicting")
    predictions = np.asarray(model.predict(x_test)).ravel()
    output_path = save_predictions(
        predictions=predictions,
        output_dir=args.output_dir,
        expected_rows=len(x_test),
    )
    print(f"Saved submission predictions to {output_path}")


if __name__ == "__main__":
    main()
