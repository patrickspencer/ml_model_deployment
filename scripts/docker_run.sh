#!/bin/bash
docker run --user="root" \
    -v $HOME/.aws/credentials:/home/jovyan/.aws/credentials:ro \
    -e AWS_PROFILE=default \
    -p 2718:3000 \
    -it model-deployment-train-model:0.1 /bin/bash
    