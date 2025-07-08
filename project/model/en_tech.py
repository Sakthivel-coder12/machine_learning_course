import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Load data and train the model once
df = pd.read_csv("model/diabetes.csv")
x = df.drop("Outcome", axis=1)
y = df["Outcome"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)

# Define and train classifiers
svm_clf = SVC(kernel="linear", C=1.0, probability=True, random_state=42)
knn_clf = KNeighborsClassifier(n_neighbors=5)
dt_clf = DecisionTreeClassifier(random_state=42)

hard_voting = VotingClassifier(
    estimators=[('svm', svm_clf), ('knn', knn_clf), ('dt', dt_clf)],
    voting="hard"
)
soft_voting = VotingClassifier(
    estimators=[('svm', svm_clf), ('knn', knn_clf), ('dt', dt_clf)],
    voting="soft"
)

# Fit both classifiers once
hard_voting.fit(x_train_scaled, y_train)
soft_voting.fit(x_train_scaled, y_train)

# Prediction function for Flask
def predict_diabetes(user_input):
    user_scaled = scaler.transform([user_input])

    prediction_soft = soft_voting.predict(user_scaled)
    prediction_hard = hard_voting.predict(user_scaled)

    result_soft = "✅Diabetic" if prediction_soft[0] == 1 else "❌ Not Diabetic"
    result_hard = "✅Diabetic" if prediction_hard[0] == 1 else "❌ Not Diabetic"

    return result_hard, result_soft
