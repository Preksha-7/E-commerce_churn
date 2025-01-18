import os
print("Running preprocessing...")
os.system("python E-commerce_churn/src/preprocessing.py")

print("Splitting data...")
os.system("python E-commerce_churn/src/train_test_split.py")

print("Training model...")
os.system("python E-commerce_churn/src/model_building.py")

print("Evaluating model...end=")
os.system("python E-commerce_churn/src/evaluate_model.py")




