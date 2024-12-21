from flask import Flask, request, jsonify
import joblib 
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('models/churn_model_info.pkl')

# Root route
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Welcome to the E-commerce Churn Prediction API!',
        'endpoints': {
            'POST /predict': 'Predict churn for a given customer profile.'
        }
    })

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON input
        data = request.json
        
        # Validate input
        if not data:
            return jsonify({'error': 'No input data provided.'}), 400
        
        # Convert input data to DataFrame
        features = pd.DataFrame(data, index=[0])
        
        # Make predictions
        prediction = model.predict(features)
        probability = model.predict_proba(features)[:, 1]
        
        return jsonify({
            'churn_prediction': int(prediction[0]),
            'churn_probability': float(probability[0])
        })
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 400

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
