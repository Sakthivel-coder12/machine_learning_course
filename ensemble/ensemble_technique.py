import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("diabetes.csv")
print("Dataset Preview:\n", df.head())

# Split into features and target
x = df.iloc[:, :-1]
y = df["Outcome"]

# Train-test split with stratification
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=42, stratify=y)

# Standardize features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Define classifiers
svm_clf = SVC(kernel="linear", C=1.0, random_state=42, probability=True)  # Soft voting needs probability=True
knn_clf = KNeighborsClassifier(n_neighbors=5)
dt_clf = DecisionTreeClassifier(random_state=42)

# Hard Voting Classifier
hard_voting = VotingClassifier(estimators=[('svm', svm_clf), ('knn', knn_clf), ('dt', dt_clf)], voting="hard")
hard_voting.fit(x_train, y_train)
y_pred_hard = hard_voting.predict(x_test)

# # Evaluation - Hard Voting
# print("🔹 Hard Voting Results:")
# print("Accuracy:", accuracy_score(y_test, y_pred_hard))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_hard))
# print("Classification Report:\n", classification_report(y_test, y_pred_hard))

# Soft Voting Classifier
soft_voting = VotingClassifier(estimators=[('svm', svm_clf), ('knn', knn_clf), ('dt', dt_clf)], voting="soft")
soft_voting.fit(x_train, y_train)
y_pred_soft = soft_voting.predict(x_test)

# # Evaluation - Soft Voting
# print("\n🔹 Soft Voting Results:")
# print("Accuracy:", accuracy_score(y_test, y_pred_soft))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_soft))
# print("Classification Report:\n", classification_report(y_test, y_pred_soft))

# Get input from user
print("\n🔍 Enter patient details to predict diabetes:")
features = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
user_input = []

for feature in features:
    value = float(input(f"Enter {feature}: "))
    user_input.append(value)

# Convert to DataFrame and standardize
user_array = np.array(user_input).reshape(1, -1)
user_scaled = scaler.transform(user_array)
user_df = pd.DataFrame([user_input], columns=features)
# Predict using soft voting classifier
prediction = soft_voting.predict(user_scaled)
pre1 = hard_voting.predict(user_scaled)
result1 = "✅ Diabetic" if pre1[0] == 1 else "❌ Not Diabetic"
print(f"\n🔎 Prediction Result: {result1}")
# Output result
result = "✅ Diabetic" if prediction[0] == 1 else "❌ Not Diabetic"
print(f"\n🔎 Prediction Result: {result}")
