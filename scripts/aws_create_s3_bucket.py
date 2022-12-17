#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
aws_create_s3_bucket.py
~~~~~~~~~~~~~~~~~~~~~~~
@Author: Patrick Spencer

Script for creating new S3 bucket
"""

import boto3

BUCKET = 'model-deployment-example-ps-v1'
REGION = 'us-west-2'
s3_session = boto3.Session().resource('s3')
s3_session.create_bucket(Bucket=BUCKET,
                         CreateBucketConfiguration={'LocationConstraint': REGION})
