from flask import Flask
from flask import jsonify
from flask import request
import pandas as pd
import numpy as np
from keras.models import model_from_json
from joblib import load
from ft_engineering import add_extra_features
from model_utils import bl_nn
from keras.wrappers.scikit_learn import KerasRegressor

app = Flask(__name__)

# load the model and standard scaler
sk_reg = KerasRegressor(
    build_fn=bl_nn(), epochs=100,
    batch_size=5, verbose=1
)

scaler = load("static/bin/final_scaler.bin")
with open("static/json/final_model_reg.json", "r") as json_file:
    loaded_model_json = json_file.read()
sk_reg.model = model_from_json(loaded_model_json)
sk_reg.model.load_weights("static/h5/final_model_reg.h5")


# define a predict function as an endpoint
@app.route("/predict", methods=["GET", "POST"])
def predict():
    data = {"success": False}
    params = request.json
    if params is None:
        params = request.args

    # if parameters are found, return a prediction
    if params is not None:
        params = params.to_dict()
        print(params)
        features = pd.DataFrame.from_records([params])
        all_features = add_extra_features(features)
        all_features_st = scaler.transform(all_features)
        pred = sk_reg.predict(all_features_st)
        data["prediction"] = str(np.exp(pred.reshape(1,)[0]))
        data["success"] = True
    return jsonify(data)

