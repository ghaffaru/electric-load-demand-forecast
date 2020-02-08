from flask import Flask, request, jsonify
from joblib import load
import pandas as pd

app = Flask(__name__)


@app.route('/')
def home():
    return 'hello'


@app.route('/api/predict/hourly', methods=['POST'])
def hourly_predict():
    hourly_model = load('model/xgb0.939.joblib')
    day = request.json['day']
    hour = request.json['hour']
    month = request.json['month']
    humidity = request.json['humidity']
    temperature = request.json['temperature']

    test = {
        'Day': day,
        'Hour': hour,
        'Month': month,
        'Relative Humidity': humidity,
        'Temperature': temperature
    }
    prediction = hourly_model.predict(pd.DataFrame(test, index=[0]))[0]
    return jsonify({
        'prediction': str(prediction)
    })


if __name__ == '__main__':
    app.run(debug=True)
