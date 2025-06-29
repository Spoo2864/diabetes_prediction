import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
import joblib

# Load dataset
data_path = r'C:\Users\spoor\OneDrive\Desktop\Python\diabetes_prediction\diabetes_prediction\diabetes.csv'
diabetes_dataset = pd.read_csv(data_path)

# Split features and target
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Standardize features
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, test_size=0.2, stratify=Y, random_state=2
)

# Train SVM
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Save model and scaler
joblib.dump(classifier, 'diabetes_prediction/svm_model.sav')
joblib.dump(scaler, 'diabetes_prediction/scaler.save')

print("✅ Model and scaler saved.")
