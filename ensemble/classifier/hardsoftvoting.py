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


def pre_diabetes(user_df):
    # Load dataset
    df = pd.read_csv("diabetes.csv")

    # Train-test split with stratification
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Define classifiers
    svm_clf = SVC(kernel="linear", C=1.0, random_state=42, probability=True)
    knn_clf = KNeighborsClassifier(n_neighbors=5)
    dt_clf = DecisionTreeClassifier(random_state=42)

    # Hard Voting Classifier
    hard_voting = VotingClassifier(
        estimators=[('svm', svm_clf), ('knn', knn_clf), ('dt', dt_clf)],
        voting="hard"
    )
    hard_voting.fit(x_train, y_train)

    # Soft Voting Classifier
    soft_voting = VotingClassifier(
        estimators=[('svm', svm_clf), ('knn', knn_clf), ('dt', dt_clf)],
        voting="soft"
    )
    soft_voting.fit(x_train, y_train)
    user_scaled = scaler.transform(user_df)

    # Predictions
    prediction_soft = soft_voting.predict(user_scaled)
    prediction_hard = hard_voting.predict(user_scaled)

    # Output results
    result_soft = "✅ Diabetic" if prediction_soft[0] == 1 else "❌ Not Diabetic"
    result_hard = "✅ Diabetic" if prediction_hard[0] == 1 else "❌ Not Diabetic"

    return result_hard,result_soft

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

# Convert input to DataFrame (to match scaler input format and avoid warnings)
user_df = pd.DataFrame([user_input], columns=features)
res1,res2 = pre_diabetes(user_df)
print(f"The hard voting is : {res1}")
print(f"The soft voting is : {res2}")