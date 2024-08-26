from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)


with open('model_pickle', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    features = [float(request.form[feature]) for feature in ['footfall', 'tempMode', 'AQ', 'USS', 'CS', 'VOC', 'RP', 'IP', 'Temperature']]
    
    # Scale features
    features = scaler.transform([features])
    
    # Predict using the model
    prediction = model.predict(features)
    
    # Determine the result
    result = "Failure predicted" if prediction[0] == 1 else "No failure predicted"
    
    return render_template('index.html', prediction_text=f'Result: {result}')


if __name__ == '__main__':
    app.run(debug=True)