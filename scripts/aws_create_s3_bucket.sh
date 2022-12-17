#!/bin/bash
aws s3api create-bucket \
    --bucket model-deployment-example-ps-v1 \
    --region us-west-2 \
    --create-bucket-configuration LocationConstraint=us-west-2