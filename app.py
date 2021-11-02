from flask import Flask, jsonify, request, render_template
import pickle
import numpy as np
import pandas as pd 
from fastai.tabular.all import *
from datetime import datetime

app = Flask(__name__)
model = load_pickle("model.pkl")
to = load_pickle("to.pkl")

Company = np.array(to.classes["Company"])
City = np.array(to.classes["City"])
start_date = datetime.strptime("01/01/2016", "%d/%m/%Y")


@app.route("/")
def home(): return render_template("index.html")


@app.route("/predict/", methods=["POST"])
def predict():
    input_val = np.array([x for x in request.form.values()])

    # make values the way we want

    temp_days = (datetime.strptime(input_val[0], "%d/%m/%Y") - start_date).days
    input_val[0] = temp_days + 42370  # 42370 is 01/01/2016
    input_val[1] = np.where(Company == input_val[1].title())[0][0]
    input_val[2] = np.where(City == input_val[2].upper())[0][0]
    input_val = input_val.astype(np.float32)

    # input_val = np.array([42540., 2., 3., 460.8, 380.4])  # test check it works

    pred_km = model.predict(np.expand_dims(input_val, axis=0))[0]
    return render_template("index.html", prediction_text=f"KM Travelled: {pred_km:.3f} km.")


if __name__ == "__main__": app.run(debug=True)