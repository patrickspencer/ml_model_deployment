from flask import Flask, request, jsonify
import json

from .utils import make_input_df, load_config, download_s3_model, make_predictions

app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return "ML predictor server is running!"


@app.route("/", methods=['POST'])
def index():
    record = json.loads(request.data)

    config = load_config('configs/predict_config.json')

    # ---------------------------------------------------------
    # Download model pickle file and input batch data from S3
    # ---------------------------------------------------------

    download_s3_model(config)

    # Setup config variables
    bucket = config["aws"]["bucket"]
    s3_model_prefix = config["aws"]["model_output_prefix"]
    s3_model_file = config["aws"]["model_file"]

    model_pkl = s3_model_prefix + '/' + s3_model_file
    data = make_input_df(record['input'])
    pred = make_predictions(model_pkl, data)
    output = {"prediction": pred.tolist()[0] if pred else 0}
    return jsonify(output)


if __name__ == '__main__':
    app.run(port=5000, debug=True, host='0.0.0.0')
