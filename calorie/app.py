from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('calorie_predictor.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    steps = data.get('steps')
    duration = data.get('duration')
    weight = data.get('weight')

    # Model expects 2D array
    prediction = model.predict([[steps, duration, weight]])[0]

    return jsonify({
        'predicted_calories': round(prediction, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)

