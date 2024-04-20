#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import pickle
from flask import Flask,request,render_template,jsonify
from sklearn.base import BaseEstimator, TransformerMixin
from werkzeug.utils import secure_filename

class GetDummies(BaseEstimator, TransformerMixin):
    def __init__(self, data_sep=',', col_name_sep='_'):
        self.dummy_prefix = []
        
    # Return self nothing else to do here
    def fit(self, X, y  = None): 
        return self
    
    # Transformer method for to return transformed data
    def transform(self, X, y = None):
        return X
        
    # to get feature names    
    def get_feature_names_out(self, input_features=None):
        return self.columns

app = Flask(__name__)
# Load the encoder from the file
#loaded_enc   = pickle.load(open("static/credit_score_multi_class_ord_encoder.pkl", "rb")) 
#loaded_le    = pickle.load(open("static/credit_score_multi_class_le.pkl", "rb"))
#loaded_dummy = pickle.load(open("static/credit_score_multi_class_dummy.pkl", "rb"))

# Load the model from the file
loaded_model = XGBClassifier()
loaded_model.load_model("static/credit_score_multi_class_xgboost_model.json")

selected_col = ['Total_EMI_per_month', 'Num_Bank_Accounts', 'Num_of_Delayed_Payment',
 'Delay_from_due_date', 'Changed_Credit_Limit', 'Num_Credit_Card',
 'Outstanding_Debt', 'Interest_Rate', 'Credit_Mix']

response_data = {
    'columns': [],
    'data': []
}

@app.route("/",methods = ["GET","POST"])
def home():
    return(render_template("index.html"))

@app.route("/upload",methods = ["GET","POST"])
def upload():
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save("static/" + filename)

    df_input = pd.read_csv("static/" + filename)
    df_test = df_input[selected_col]
    x_mapping = {"Bad":0, "Good":1, "Standard":2}
    df_test['Credit_Mix'] = df_test['Credit_Mix'].map(x_mapping)

    y_test_pred = loaded_model.predict(df_test)
    y_mapping = {1: "Poor", 2: "Standard", 0:"Good"}
    pred_cat = [y_mapping[value] for value in y_test_pred]

    df_final = df_input.copy()
    df_final['Credit Score'] = pred_cat
    response_data = {
        'columns': df_final.columns.tolist(),
        'data': df_final.values.tolist()
    }
    return jsonify(response_data)
    

if __name__ == "__main__":
    app.run()
