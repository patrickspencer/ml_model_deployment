#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
predict.py
~~~~~~~~~~
@Author: Patrick Spencer

Model inference from saved model file

Example usage:
    predict.py predict_config.json

Must have input config file
"""
import sys
import pandas as pd
from numpy import savetxt
from utils import (
    load_config, load_pickled_model, s3_download_file,
    create_file_suffix_dt,
    s3_upload_file,
    logger,
    make_predictions,
    download_s3_model
)


def predict() -> None:
    """Predict function. Loads config file with model pickle file name
    and data input file
    """
    try:
        config_file = sys.argv[1]
    except IndexError:
        print("Error: Please specify json config file: `python3 predict.py config.json`")
        sys.exit()

    config = load_config(config_file)

    # ---------------------------------------------------------
    # Download model pickle file and input batch data from S3
    # ---------------------------------------------------------

    download_s3_model(config)

    # Download predictor data from aws s3 bucket
    bucket = config["aws"]["bucket"]
    s3_model_prefix = config["aws"]["model_output_prefix"]
    s3_model_file = config["aws"]["model_file"]
    model_pkl = s3_model_prefix + '/' + s3_model_file
    s3_predictor_input_data = config["aws"]["batch_input_data"]

    predictor_data_folder = 'predictor_input'
    predictor_data_file = 'data.csv'
    predictor_input_full = predictor_data_folder + '/' + predictor_data_file

    s3_download_file(predictor_input_full, bucket, s3_predictor_input_data)

    # ---------------------------------------------------------
    # Read in classifier
    # ---------------------------------------------------------
    try:
        data = pd.read_csv(predictor_input_full)
    except FileNotFoundError:
        print(
            f"Error: Cannot read input data file. File name `{predictor_input_full}`")

    pred = make_predictions(model_pkl, data)
    print('data:', data.columns)

    file = 'predictor_output_' + create_file_suffix_dt() + '.csv'
    output_file = 'predictor_output/' + file

    output = 'saving output to ' + output_file
    savetxt(output_file, pred, delimiter=',')

    # ---------------------------------------------------------
    # Upload output data to S3
    # ---------------------------------------------------------
    s3_upload_file(output_file,
                   bucket,
                   object_name=output_file)

    log_output = config["predict_logs_output_folder"]
    log_file = 'predictor_log' + '_1_' + '.txt'
    logger(output, log_output+'/' + log_file)
    # model_output_file = 'model_' + suffix + '.pkl'
    # logger(output, )


if __name__ == "__main__":
    predict()
