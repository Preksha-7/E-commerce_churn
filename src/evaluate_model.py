import tensorflow as tf
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

model = tf.keras.models.load_model('models/churn_model.h5')

X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv')

y_pred = (model.predict(X_test)>0.5).astype("int32")

print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Classification report:\n", classification_report(y_test,y_pred))