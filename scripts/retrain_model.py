import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from sklearn.preprocessing import MinMaxScaler
import os

def preprocess_data(df):
    # Fill missing values
    df.fillna(df.select_dtypes(include=['number']).mean(), inplace=True)  
    df.fillna(df.select_dtypes(include=['object']).mode().iloc[0], inplace=True) 
    
    df = pd.get_dummies(df, columns=['PreferedOrderCat', 'MaritalStatus'], drop_first=True)
    
    # Scale numeric columns
    numeric_cols = ['Tenure', 'WarehouseToHome', 'NumberOfDeviceRegistered', 
                    'SatisfactionScore', 'DaySinceLastOrder', 'CashbackAmount']
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df


# Load new data
data_path = 'E-commerce_churn/data/latest_customer_data.csv'
df = pd.read_csv(data_path)

# Preprocess the data
df = preprocess_data(df)

# Train-Test Split
X = df.drop('Churn', axis=1)  # Features
y = df['Churn']  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Retrain the Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

# Ensure the directory for saving the model exists
model_dir = 'models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Save the model
joblib.dump(model, os.path.join(model_dir, 'churn_model.pkl'))

print("Model saved as 'churn_model.pkl' in the 'models' directory.")
