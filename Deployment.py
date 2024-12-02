import os
print(os.getcwd())
import sklearn
print(sklearn.__version__)
from flask import Flask, request, jsonify
import joblib
import numpy as np
model = joblib.load(r'C:\Users\jilow\Machine_learning_assignment1\decision_tree_model_best_tuned.pk2')
# Initialize the Flask app
app = Flask(__name__)

# Load the saved model
model = joblib.load(r'C:\Users\jilow\Machine_learning_assignment1\decision_tree_model_best_tuned.pk2')

# Define a route for the root
@app.route('/')
def home():
    return "Welcome to the Genetic Disorder Prediction API!"

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        data = request.get_json()  # Assuming the input is sent as a JSON object
        
        # Extract the features from the data
        features = np.array([data['Patient Age'],
                             data['Blood cell count (mcL)'],
                             data['Mother\'s age'],
                             data['Father\'s age'],
                             data['Test 1'],
                             data['Test 2'],
                             data['Test 3'],
                             data['Test 4'],
                             data['Test 5'],
                             data['No. of previous abortion'],
                             data['White Blood cell count (thousand per microliter)'],
                             data['Symptom 1'],
                             data['Symptom 2'],
                             data['Symptom 3'],
                             data['Symptom 4'],
                             data['Symptom 5']])
        
        # Reshape the input data to match the model input format
        features = features.reshape(1, -1)
        
        # Predict the class using the model
        prediction = model.predict(features)
        
        # Return the prediction as a response
        return jsonify({'prediction': int(prediction[0])})
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

