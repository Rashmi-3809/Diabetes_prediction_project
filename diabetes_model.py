import pandas as pd

# Load dataset
data = pd.read_csv(r"C:/Users/Dell/OneDrive/Desktop/Diabetes_Prediction_Project/data/diabetes.csv")  # "../data" ka matlab parent folder ke data folder

# Check first 5 rows
print(data.head())
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (important for Logistic Regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)
 
# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
import pickle
import os

# Absolute path
model_path = r"C:\Users\Dell\OneDrive\Desktop\Diabetes_Prediction_Project\models\diabetes_model.pkl"

# Ensure folder exists
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# Save model
with open(model_path, "wb") as f:
    pickle.dump(model, f)