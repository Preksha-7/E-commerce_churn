import tensorflow as tf
import pickle
model = tf.keras.models.load_model('E-commerce_churn/models/churn_model.h5')

model_info = {
    "model_architecture": model.to_json(),
    "weights": model.get_weights()
}

with open('E-commerce_churn/models/churn_model_info.pkl', 'wb') as f:
    pickle.dump(model_info, f)

print("Model information saved as churn_model_info.pkl")
