from flask import *
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import json, time
import joblib
import re
import os
from flask_cors import CORS

# save np.load
np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

encoder = LabelEncoder()
encoder.classes_ = np.load('classes.npy')

filename = 'xgboost_model.joblib'
model = joblib.load(filename)
precautions = pd.read_csv('disease_precautions.csv')
info = pd.read_csv('28_disease_info_translated.csv')


app = Flask(__name__)
CORS(app)

@app.route('/', methods = ['GET'])
def home_page():
    data_set = {'Page': 'Home', 'Message': "Let's get started and send me your symptoms", 'Timestamp': time.time()}
    json_dump = json.dumps(data_set)
    return json_dump

@app.route('/predict/', methods=['GET'])
def request_page():
    symptoms = request.args.get('symptoms') # /predict/?symptoms=symptoms
    symptoms= re.findall('\d', symptoms)
    symptoms= list(map(int,symptoms))
    if  sum(symptoms)<3:
        data_set = {'warning' : 'For a better result, please choose at least 3 symptoms'}

    else:
        disease_index = model.predict([symptoms])[0]
        confidence = str(int(model.predict_proba([symptoms])[0].max()*100))+'%'
        disease = encoder.classes_[disease_index]

        data_set = {'prediction': disease,
                    "confidence": confidence,
                    "precaution_1": precautions[precautions.Disease == disease].iloc[0][2],
                    "precaution_2": precautions[precautions.Disease == disease].iloc[0][3],
                    "precaution_3": precautions[precautions.Disease == disease].iloc[0][4],
                    "precaution_4": precautions[precautions.Disease == disease].iloc[0][5],
                    "prediction_in_arabic": precautions[precautions.Disease == disease].iloc[0][6],
                    "precaution_1_in_arabic": precautions[precautions.Disease == disease].iloc[0][7],
                    "precaution_2_in_arabic": precautions[precautions.Disease == disease].iloc[0][8],
                    "precaution_3_in_arabic": precautions[precautions.Disease == disease].iloc[0][9],
                    "precaution_4_in_arabic": precautions[precautions.Disease == disease].iloc[0][10],
                    
                    'Overview': info[info.disease == disease]['Overview'].values.tolist(),
                    'Link': info[info.disease == disease]['link'].values.tolist(),
                    'Causes': info[info.disease == disease]['Causes'].values.tolist(),
                    'Risk_Factors': info[info.disease == disease]['Risk factors'].values.tolist(),
                   
                    'Overview_in_arabic': info[info.disease == disease]['Overview_in_arabic'].values.tolist(),
                    'Causes_in_arabic': info[info.disease == disease]['Causes_in_arabic'].values.tolist(),
                    'Risk_Factors_in_arabic': info[info.disease == disease]['Risk_factors_in_arabic'].values.tolist()}
                     
    json_dump = json.dumps(data_set)
    return json_dump

if __name__=='__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
