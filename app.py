from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import json

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = None

def load_model():
    """Load the trained TensorFlow model"""
    global model
    try:
        model = tf.keras.models.load_model('models/churn_model.h5')
        print("Model loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

# Feature list required for prediction
REQUIRED_FEATURES = [
    'Tenure', 'WarehouseToHome', 'HourSpendOnApp', 'NumberOfDeviceRegistered',
    'SatisfactionScore', 'NumberOfAddress', 'Complain', 'OrderAmountHikeFromlastYear',
    'CouponUsed', 'OrderCount', 'DaySinceLastOrder', 'CashbackAmount',
    'PreferedOrderCat_Fashion', 'PreferedOrderCat_Grocery', 'PreferedOrderCat_Mobile',
    'MaritalStatus_Married', 'MaritalStatus_Single'
]

@app.route('/', methods=['GET'])
def home():
    """Root endpoint with API information"""
    return jsonify({
        'status': 'success',
        'message': 'Welcome to the E-commerce Churn Prediction API!',
        'model_loaded': model is not None,
        'endpoints': {
            'POST /predict': 'Predict churn for a given customer profile',
            'GET /health': 'Check API health status'
        },
        'required_features': REQUIRED_FEATURES
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Make churn predictions based on input data"""
    try:
        # Ensure model is loaded
        if model is None:
            if not load_model():
                return jsonify({'error': 'Model not loaded. Try again later.'}), 500

        # Parse JSON input
        data = request.json
        
        # Validate input
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Create DataFrame with all required features
        input_data = {}
        for feature in REQUIRED_FEATURES:
            if feature in data:
                input_data[feature] = data[feature]
            else:
                return jsonify({'error': f'Missing required feature: {feature}'}), 400
        
        # Convert to DataFrame
        features_df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction_prob = model.predict(features_df)[0][0]
        prediction = int(prediction_prob >= 0.5)
        
        return jsonify({
            'status': 'success',
            'churn_prediction': prediction,
            'churn_probability': float(prediction_prob),
            'message': 'Customer likely to churn' if prediction == 1 else 'Customer likely to stay'
        })
    
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

if __name__ == '__main__':
    # Load model at startup
    load_model()
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
    
    
@app.route('/sample', methods=['GET'])
def sample():
    """Return a sample input for testing the prediction endpoint"""
    sample_data = {
        'Tenure': 0.5,
        'WarehouseToHome': 0.3,
        'HourSpendOnApp': 2.5,
        'NumberOfDeviceRegistered': 3,
        'SatisfactionScore': 4,
        'NumberOfAddress': 2,
        'Complain': 0,
        'OrderAmountHikeFromlastYear': 0.15,
        'CouponUsed': 2,
        'OrderCount': 5,
        'DaySinceLastOrder': 0.2,
        'CashbackAmount': 0.1,
        'PreferedOrderCat_Fashion': 1,
        'PreferedOrderCat_Grocery': 0,
        'PreferedOrderCat_Mobile': 0,
        'MaritalStatus_Married': 1,
        'MaritalStatus_Single': 0
    }
    
    return jsonify({
        'status': 'success',
        'sample_data': sample_data,
        'instructions': 'Use this sample data with the POST /predict endpoint'
    })