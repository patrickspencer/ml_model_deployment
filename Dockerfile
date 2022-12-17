FROM jupyter/scipy-notebook

ADD lib ./lib
ADD scripts ./scripts
ADD configs ./configs
ADD data ./data
ADD log_output ./log_output
ADD model_output ./model_output
ADD predictor_output ./predictor_output
ADD predictor_input ./predictor_input

COPY requirements.txt ./requirements.txt

WORKDIR ./

RUN pip install -r requirements.txt
# RUN sh start_predictor_api.sh