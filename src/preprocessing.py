import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

file_path = 'data/data_ecommerce_customer_churn.csv'
df = pd.read_csv(file_path)

df.fillna(df.select_dtypes(include=['number']).mean(), inplace=True)

df.fillna(df.select_dtypes(include=['object']).mode().iloc[0], inplace=True)

#print("Missing values after imputation:")
#print(df.isnull().sum())
sns.countplot(x='Churn', data=df)
plt.title("Distribution of Churn")
#plt.show()
#print(df.describe())

df = pd.get_dummies(df, columns=['PreferedOrderCat', 'MaritalStatus'], drop_first=True)
#print(df.head())

numeric_cols = ['Tenure', 'WarehouseToHome', 'NumberOfDeviceRegistered', 
                'SatisfactionScore', 'DaySinceLastOrder', 'CashbackAmount']


scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

#print(df[numeric_cols].head())


X = df.drop('Churn', axis=1)
y = df['Churn']

#print(f"Features shape: {X.shape}")
#print(f"Target shape: {y.shape}")

# Save to processed data folder
processed_path = 'data/processed_ecommerce_churn.csv'
df.to_csv(processed_path, index=False)

print("Data preprocessing complete. File saved at:", processed_path)
