import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os

data = pd.read_csv('data/processed_ecommerce_churn.csv')
X = data.drop('Churn', axis = 1)
y = data['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify= y)

print("Class distribution before SMOTE:")
print(y_train.value_counts())

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
print("Class distribution after SMOTE:")
print(y_train.value_counts())

# Ensure data directory exists
if not os.path.exists('data'):
    os.makedirs('data')

X_train.to_csv('data/X_train.csv', index = False)
X_test.to_csv('data/X_test.csv', index = False)
y_train.to_csv('data/y_train.csv', index = False)
y_test.to_csv('data/y_test.csv', index = False)

print('Data split complete and saved')