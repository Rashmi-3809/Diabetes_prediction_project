import pickle
import numpy as np

# Load trained model
with open("models/diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

# Sample input [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
sample_input = np.array([[2, 140, 80, 30, 0, 28.5, 0.45, 35]])

# Predict
prediction = model.predict(sample_input)

# Output
if prediction[0] == 1:
    print("Patient is likely to have diabetes")
else:
    print("Patient is likely NOT to have diabetes")