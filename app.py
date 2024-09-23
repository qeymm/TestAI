from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

# Load the saved model
model = joblib.load('insurance_model.pkl')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.json
    age = data['age']
    gender = 0 if data['gender'] == 'Male' else 1
    country = 0 if data['country'] == 'Indonesia' else 1
    insurance_type = 0 if data['insurance_type'] == 'Health' else 1

    # Prepare the input for the model
    input_data = np.array([[age, gender, country, insurance_type]])

    # Predict the best plan (0 = Basic, 1 = Silver, 2 = Gold)
    prediction = model.predict(input_data)

    # Convert the prediction to a readable string
    plans = {0: 'Basic', 1: 'Silver', 2: 'Gold'}
    best_plan = plans[prediction[0]]

    # Return the result as JSON
    return jsonify({'best_plan': best_plan})

if __name__ == '__main__':
    app.run(debug=True)
