from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import json
import pickle
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Model file paths
h5_model_path = os.path.join(current_dir, 'models', 'churn_model.h5')
pkl_model_path = os.path.join(current_dir, 'models', 'churn_model_info.pkl')


# Global model variable
model = None

def load_model():
    """Load the trained TensorFlow model"""
    global model
    try:
        # Try to load the .h5 model first
        if os.path.exists(h5_model_path):
            model = tf.keras.models.load_model(h5_model_path)
            print("Model loaded successfully from .h5 file")
            return True
        # Fall back to the pickle file
        elif os.path.exists(pkl_model_path):
            with open(pkl_model_path, 'rb') as f:
                model_info = pickle.load(f)
            
            # Reconstruct the model from the saved architecture and weights
            model = tf.keras.models.model_from_json(model_info["model_architecture"])
            model.set_weights(model_info["weights"])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            print("Model loaded successfully from pickle file")
            return True
        else:
            print("No model file found")
            return False
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

if __name__ == '__main__':
    model_loaded = load_model()
    
    # Get and print input shape safely
    if model_loaded and model is not None:
        input_shape = model.input_shape
        print(f"Model expects input shape: {input_shape}")
        
        # Check if number of input features matches REQUIRED_FEATURES
        expected_features = input_shape[1] if len(input_shape) > 1 else None
        if expected_features and expected_features != len(REQUIRED_FEATURES):
            print(f"⚠️ WARNING: Model expects {expected_features} features but REQUIRED_FEATURES has {len(REQUIRED_FEATURES)}.")
        else:
            print("✅ Feature count matches model input shape.")
    else:
        print("❌ Model is not loaded. Cannot determine input shape.")

    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)

@app.route('/', methods=['GET'])
def home():
    """Root endpoint with API information"""
    return jsonify({
        'status': 'success',
        'message': 'Welcome to the E-commerce Churn Prediction API!',
        'model_loaded': model is not None,
        'endpoints': {
            'POST /predict': 'Predict churn for a given customer profile',
            'GET /health': 'Check API health status',
            'GET /sample': 'Get sample data for testing'
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
    
    
# In app.py, add this debugging function
def get_model_input_features():
    """Determine the exact feature list required by the model"""
    model = tf.keras.models.load_model('models/churn_model.h5')
    input_shape = model.layers[0].input_shape
    expected_features = input_shape[1] if len(input_shape) > 1 else input_shape[0][1]
    print(f"Model expects {expected_features} input features")
    
    # Now let's attempt to determine what these features are by looking at training data
    try:
        train_data = pd.read_csv('data/X_train.csv')
        feature_columns = train_data.columns.tolist()
        print(f"Training data has {len(feature_columns)} features: {feature_columns}")
        return feature_columns
    except Exception as e:
        print(f"Could not determine feature columns from training data: {str(e)}")
        return None

# Call this function when loading the app
if __name__ == '__main__':
    # Load model at startup
    load_model()
    # Get expected features
    model_features = get_model_input_features()
    if model_features and len(model_features) != len(REQUIRED_FEATURES):
        print("WARNING: Feature mismatch! Updating REQUIRED_FEATURES to match model expectations.")
        REQUIRED_FEATURES = model_features
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
    