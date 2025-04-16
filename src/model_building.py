import tensorflow as tf
import pandas as pd
import numpy as np
import os

X_train = pd.read_csv('data/X_train.csv')
X_test = pd.read_csv('data/X_test.csv')
y_train = pd.read_csv('data/y_train.csv').squeeze()
y_test = pd.read_csv('data/y_test.csv').squeeze()

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation = 'relu', input_shape = (X_train.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])

model.compile(optimizer ='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
history = model.fit(X_train, y_train,
                    epochs=20,
                    batch_size=32,
                    validation_data = (X_test, y_test),
                    verbose = 2
                    )

# Ensure the models directory exists
if not os.path.exists('models'):
    os.makedirs('models')

model.save('models/churn_model.h5')
print('Model training complete and saved')