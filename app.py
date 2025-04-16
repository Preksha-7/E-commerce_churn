from flask import Flask, request, jsonify
from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import json
import pickle

# Initialize Flask app
app = Flask(__name__)

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Model file paths
h5_model_path = os.path.join(current_dir, 'models', 'churn_model.h5')
pkl_model_path = os.path.join(current_dir, 'models', 'churn_model_info.pkl')

# Global model and feature variables
model = None
REQUIRED_FEATURES = [
    'Tenure', 'WarehouseToHome', 'HourSpendOnApp', 'NumberOfDeviceRegistered',
    'SatisfactionScore', 'NumberOfAddress', 'Complain', 'OrderAmountHikeFromlastYear',
    'CouponUsed', 'OrderCount', 'DaySinceLastOrder', 'CashbackAmount',
    'PreferedOrderCat_Fashion', 'PreferedOrderCat_Grocery', 'PreferedOrderCat_Mobile',
    'MaritalStatus_Married', 'MaritalStatus_Single'
]

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

def get_model_input_features():
    """Determine the exact feature list required by the model"""
    if model is None:
        return None
        
    input_shape = model.input_shape
    expected_features = input_shape[1] if len(input_shape) > 1 else None
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
    sample_data = {}
    for feature in REQUIRED_FEATURES:
        # Generate reasonable sample values
        if feature.startswith('PreferedOrderCat_') or feature.startswith('MaritalStatus_'):
            sample_data[feature] = 1 if feature.endswith('_Fashion') or feature.endswith('_Married') else 0
        elif feature == 'Complain':
            sample_data[feature] = 0
        elif feature in ['NumberOfDeviceRegistered', 'SatisfactionScore', 'NumberOfAddress', 'CouponUsed', 'OrderCount']:
            sample_data[feature] = 2
        else:
            sample_data[feature] = 0.5
            
    return jsonify({
        'status': 'success',
        'sample_data': sample_data,
        'instructions': 'Use this sample data with the POST /predict endpoint'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Make churn predictions based on input data"""
    global REQUIRED_FEATURES
    
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
        
        # Check if we need to adjust features based on model input shape
        expected_features = model.input_shape[1] if len(model.input_shape) > 1 else None
        
        if expected_features is not None and expected_features != len(REQUIRED_FEATURES):
            # We need to adjust features to match the model
            if expected_features < len(REQUIRED_FEATURES):
                # Try to use training data to identify correct features
                model_features = get_model_input_features()
                if model_features:
                    # Use only the features the model expects
                    features_df = features_df[model_features]
                else:
                    # Fall back to using first 'expected_features' from the input
                    features_df = features_df.iloc[:, :expected_features]
            else:
                # Model expects more features than provided - this shouldn't happen
                return jsonify({'error': f'Model expects {expected_features} features but only {len(REQUIRED_FEATURES)} were provided'}), 400
        
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
    
    # Get and print input shape safely
    if model is not None:
        input_shape = model.input_shape
        print(f"Model expects input shape: {input_shape}")
        
        # Check if number of input features matches REQUIRED_FEATURES
        expected_features = input_shape[1] if len(input_shape) > 1 else None
        if expected_features and expected_features != len(REQUIRED_FEATURES):
            print(f"⚠️ WARNING: Model expects {expected_features} features but REQUIRED_FEATURES has {len(REQUIRED_FEATURES)}.")
            
            # Try to get correct feature list from training data
            model_features = get_model_input_features()
            if model_features:
                print("✅ Updating REQUIRED_FEATURES to match model input shape.")
                REQUIRED_FEATURES = model_features
        else:
            print("✅ Feature count matches model input shape.")
    else:
        print("❌ Model is not loaded. Cannot determine input shape.")
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
    
@app.route('/ui')
def ui():
    """Serve the static HTML file"""
    return send_from_directory('static', 'index.html')

# Update your Flask app initialization to enable CORS
# Add this import at the top
from flask_cors import CORS

# Then modify your Flask app initialization
app = Flask(__name__)
CORS(app)