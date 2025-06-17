#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import pickle
from Models import load_models

app = Flask(__name__)

# Load the trained models (Naive Bayes and Perceptron)
naive_bayes_model, perceptron_model = load_models()

# Serve the index.html page from the static folder
@app.route('/')
def home():
    return render_template('index.html')

# API to make predictions using Naive Bayes and Perceptron
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get the data from the frontend
    features = np.array(data['features']).reshape(1, -1)  # Reshape for prediction

    # Naive Bayes prediction (binary output: 0 or 1)
    nb_prediction = naive_bayes_model.predict(features)[0]
    
    # Perceptron prediction (binary output: 0 or 1)
    perceptron_prediction = perceptron_model.predict(features)[0]

    # Return both predictions in the response
    response = {
        'naive_bayes_prediction': int(nb_prediction),
        'perceptron_prediction': int(perceptron_prediction)
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=10000, debug=True)

