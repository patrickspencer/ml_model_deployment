#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train.py
~~~~~~~~
@Author: Patrick Spencer

Trains the model

Example usage:
    train.py train_config.json

Must have input config file
"""

import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
from utils import (
    logger, save_model, load_config, s3_upload_file,
    create_log_output_file_full, create_log_output_file,
    create_file_suffix
)


def train() -> None:
    """Trains new model from config file. Takes config file as input"""
    try:
        config_file = sys.argv[1]
    except IndexError:
        print("Error: Please specify json config file: `sh train.sh config.json`")
        sys.exit()

    config = load_config(config_file)

    # Load Data
    try:
        df = pd.read_csv(config["training_data_input"])
    except FileNotFoundError:
        print(
            f"Error: Cannot read config file {config['training_data_input']}")
        sys.exit()

    # Check data is non-empty
    try:
        assert df.head(5).shape and df.head(5).shape[0] == 5
    except AssertionError:
        print("Training data seems to be empty")
        sys.exit()

    # ---------------------------------------------------------
    # Model Training
    # ---------------------------------------------------------

    df_X = df.drop("y", axis=1)
    df_label = df["y"]

    numeric_features = ["x1", "x2", "x4", "x5"]
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")),
               ("scaler", StandardScaler())]
    )

    categorical_features = ["x3", "x6", "x7"]
    categorical_transformer = OneHotEncoder(
        handle_unknown="infrequent_if_exist")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    clf = Pipeline(
        steps=[("preprocessor", preprocessor),
               ("classifier", LogisticRegression(
                max_iter=config['model_params']['max_iter'],
                penalty=config['model_params']['penalty'],
                dual=config['model_params']['dual'],
                C=config['model_params']['C'],
                tol=config['model_params']['tol'],
                solver=config['model_params']['solver'],
                multi_class=config['model_params']['multi_class']
                ))]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        df_X,
        df_label,
        random_state=config['model_params']['random_state']
    )

    print('Starting model training')
    clf.fit(X_train, y_train)
    print("Finished model training.\nModel score: %.3f" %
          clf.score(X_test, y_test))

    tprobs = clf.predict_proba(X_test)[:, 1]

    # ---------------------------------------------------------
    # Load config variables
    # ---------------------------------------------------------

    model_version = config["training_model_output_version"]
    log_output_folder = config["training_logs_output_folder"]
    model_output_folder = config["training_model_output_folder"]

    # ---------------------------------------------------------
    # Save Pickle File
    # ---------------------------------------------------------

    suffix = create_file_suffix(model_version)
    model_output_file = 'model_' + suffix + '.pkl'
    model_output_file_full = model_output_folder + '/' + model_output_file
    save_model(clf, model_output_file_full)

    # ---------------------------------------------------------
    # Make logs
    # ---------------------------------------------------------

    log_output = classification_report(y_test, clf.predict(X_test))
    log_output += 'Confusion matrix:' + '\n'
    log_output += str(confusion_matrix(y_test, clf.predict(X_test))) + '\n'
    log_output += f'AUC: {roc_auc_score(y_test, tprobs)}' + '\n'
    log_output += 'Saving pickle file to: ' + model_output_file_full
    log_output_file = create_log_output_file(suffix)
    log_output_file_full = create_log_output_file_full(
        log_output_folder, suffix)
    logger(log_output, log_output_file_full)

    print(config["model_params"])

    # ---------------------------------------------------------
    # Upload to S3
    # ---------------------------------------------------------

    bucket = config["aws"]["bucket"]
    s3_model_prefix = config["aws"]["model_output_prefix"]
    s3_log_prefix = config["aws"]["log_output_prefix"]
    s3_model_prefix_full = s3_model_prefix + '/' + model_output_file
    s3_log_prefix_full = s3_log_prefix + '/' + log_output_file
    print('uploading model file to:', bucket + '/' + s3_model_prefix_full)
    print('uploading log file to:', bucket + '/' + s3_log_prefix_full)

    s3_upload_file(model_output_file_full,
                   bucket,
                   object_name=s3_model_prefix_full)
    s3_upload_file(log_output_file_full, bucket,
                   object_name=s3_log_prefix_full)


if __name__ == "__main__":
    train()
