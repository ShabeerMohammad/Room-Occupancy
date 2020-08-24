# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the logistic regression classifier model
filename = 'logistic-regression.pkl'
LogisticModel = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        Temperature = float(request.form['Temperature'])
        Humidity = float(request.form['Humidity'])
        Light = float(request.form['Light'])
        CO2 = float(request.form['CO2'])
        HumidityRatio = float(request.form['HumidityRatio'])
        
        data = np.array([[Temperature, Humidity, Light, CO2, HumidityRatio]])
        my_prediction = LogisticModel .predict(data)
        
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)