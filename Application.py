import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
import pickle

app = Flask(__name__)
model=pickle.load(open('MedicalCostPredict.pkl','rb'))
medical=pd.read_csv('cleanedMedicalRecord.csv')
print(model)
@app.route('/')
def index():
    sex=medical['sex'].unique()
    smoker=medical['smoker'].unique()
    region=medical['region'].unique()

    return render_template('index.html',sex=sex,smoker=smoker,region=region)

@app.route('/predict',methods=['POST'])
def predict():
    age=int(request.form.get('age'))
    sex=request.form.get('gender')
    bmi=float(request.form.get('bmi'))
    children=int(request.form.get('children'))
    smoker = request.form.get('smoker')
    region=request.form.get('region')
    predction = model.predict(pd.DataFrame([[age,sex,bmi, children, smoker, region]], columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region']))
    print(predction[0] )
    return str(np.round(predction[0],2))



if __name__=='__main__':
    app.run(debug=True)