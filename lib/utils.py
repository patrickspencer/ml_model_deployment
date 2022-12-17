#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
utils.py
~~~~~~~~
@Author: Patrick Spencer

Common utility functions
"""

import os
from botocore.exceptions import ClientError
import boto3
import sys
import logging
import json
import pickle
from datetime import datetime
import pandas as pd


def load_config(config_file):
    """Loads json config from input file"""
    try:
        open(config_file, "rb")
    except IOError:
        print(
            f"Error: Config file {config_file} does not appear to exist.")
        sys.exit()

    with open(config_file, "rb") as jsonfile:
        config = json.load(jsonfile)

    return config


def create_log_output_file(suffix):
    return 'training_log_' + suffix + '.log'


def create_log_output_file_full(output_folder, suffix):
    file_name = create_log_output_file(suffix)
    return output_folder + '/' + file_name


def logger(output, output_file_full):
    """Make basic logger
    output_file_full is folder + filename
    """
    print(output)
    logging.basicConfig(filename=output_file_full,
                        format='%(message)s',
                        filemode='w')
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    log.info(output)


def save_model(clf, output_file):
    """Save classifier to specified output folder"""
    pickle.dump(clf, open(output_file, 'wb'))
    print('Saving pickle file to:', output_file)


def load_pickled_model(model_pickle_file):
    """Loads pickeled classifier from input file"""
    try:
        open(model_pickle_file, "rb")
    except IOError:
        print(
            f"Error: Model file {model_pickle_file} does not appear to exist.")
        sys.exit()

    with open(model_pickle_file, "rb") as model_file:
        model = pickle.load(model_file)

    return model


def s3_download_file(file_name, bucket, object_name=None):
    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        # download_file('BUCKET_NAME', 'OBJECT_NAME', 'FILE_NAME')
        response = s3_client.download_file(bucket, object_name, file_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True


def s3_upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True


def create_file_suffix_dt():
    return datetime.now().isoformat()


def create_file_suffix(model_version):
    return model_version + '_' + create_file_suffix_dt()


def save_data_to_file(data, file_output):
    # with open("output.txt", "w") as txt_file:
    # for line in data:
    #     # works with any number of elements in a line
    #     txt_file.write(" ".join(line) + "\n")
    print(data)
    print(file_output)


def make_input_df(data):
    """makea dataframe from input data

    input data is a string. This is an example:
    "1.503407,15.165184,Monday,0.848177,112.883872,New Mexico,toyota"

    This will be used for the API

    Output is a dataframe where columns are x1 ... x7
    first row is the input data
    """
    cols = ['x' + str(i) for i in list(range(1, 8))]
    b = data.split(',')
    pairs = list(zip(cols, b))
    d = {x: [y] for x, y in pairs}
    df = pd.DataFrame(d)
    return df


def make_predictions(model_pkl, data):
    # load the model from disk
    clf = load_pickled_model(model_pkl)
    pred = clf.predict_proba(data)[:, 1]
    return pred


def download_s3_model(config):
    predict_data_input = config["predict_data_input"]
    # predict_model_pickle_file = config["predict_model_pickle_file"]
    bucket = config["aws"]["bucket"]
    s3_model_prefix = config["aws"]["model_output_prefix"]
    s3_model_file = config["aws"]["model_file"]
    local_file = 'model_output/' + s3_model_file
    model_pkl = s3_model_prefix + '/' + s3_model_file

    # Download model from aws s3 bucket
    s3_download_file(local_file, bucket, model_pkl)
