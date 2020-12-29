import uvicorn  ##ASGI 
from fastapi import FastAPI
import pickle
import pandas as pd
import numpy as np
import BankNotes as BankNote

app=FastAPI()
pickle_in=open('model.pkl', 'rb')
classifier=pickle.load(pickle_in)

@app.get('/')
def index():
    return {'message': 'Hello, World'}

@app.post('/predict')
def predict_bankNote(data:BankNote):
	data=data.dict()
	variance=data['variance']
	skewness=data['skewness']
	curtosis=data['curtosis']
	entropy=data['entropy']

	prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
	if prediction[0] >0.7:
		prediction='Fake Note'
	else:
		prediction='It\'s a Bank Note'
	return {'Prediction':prediction }

if __name__ == '__main__':
	uvicorn.run(app,host='127.0.0.1',port=8000)

#uvicorn app:app --reload
