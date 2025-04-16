import tensorflow as tf
import pickle
import os

model = tf.keras.models.load_model('models/churn_model.h5')

model_info = {
    "model_architecture": model.to_json(),
    "weights": model.get_weights()
}

# Ensure models directory exists
if not os.path.exists('models'):
    os.makedirs('models')

with open('models/churn_model_info.pkl', 'wb') as f:
    pickle.dump(model_info, f)

print("Model information saved as churn_model_info.pkl")