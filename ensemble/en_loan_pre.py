import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import class_weight

def predict_loan_default(user_df):
    try:
        # Load dataset with correct encoding and memory options
        df = pd.read_csv("loan.csv", encoding='latin-1', low_memory=False)
    except FileNotFoundError:
        raise FileNotFoundError("Loan dataset not found. Ensure 'loan.csv' exists in the working directory.")
    except Exception as e:
        raise Exception(f"Error loading CSV: {e}")
    
    # Filter only relevant loan statuses
    df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]
    
    # Convert percentage columns to numeric if they are object type
    if df['int_rate'].dtype == 'O':
        df['int_rate'] = df['int_rate'].str.rstrip('%').astype(float)
    if df['revol_util'].dtype == 'O':
        df['revol_util'] = df['revol_util'].str.rstrip('%').replace('', np.nan).astype(float)
    
    # Select numerical features for prediction
    numerical_features = [
        'loan_amnt', 'funded_amnt', 'int_rate', 'annual_inc', 'dti',
        'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'revol_bal', 'revol_util', 'total_acc'
    ]
    # Check for missing columns
    missing_cols = set(numerical_features) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in CSV: {missing_cols}")
    
    # Prepare features and target
    X = df[numerical_features].fillna(0)  # Handle missing values
    
    # Encode loan_status: "Fully Paid"=0, "Charged Off"=1
    le = LabelEncoder()
    y = le.fit_transform(df['loan_status'])
    
    # Check class balance
    class_counts = np.bincount(y)
    if np.min(class_counts) < 2:
        raise ValueError("Insufficient samples for one of the classes after filtering. "
                         "Ensure both 'Fully Paid' and 'Charged Off' exist in your data.")
    
    # Train-test split with stratification
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Define classifiers with class_weight for imbalance
    svm_clf = SVC(kernel="linear", C=1.0, random_state=42, probability=True, class_weight='balanced')
    knn_clf = KNeighborsClassifier(n_neighbors=5)
    dt_clf = DecisionTreeClassifier(random_state=42, class_weight='balanced')

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
    
    # Scale user input
    user_scaled = scaler.transform(user_df)

    # Predictions
    prediction_soft = soft_voting.predict(user_scaled)
    prediction_hard = hard_voting.predict(user_scaled)

    # Output results (assuming 1 = default risk, 0 = likely to pay)
    result_soft = "⚠️ High Default Risk" if prediction_soft[0] == 1 else "✅ Low Default Risk"
    result_hard = "⚠️ High Default Risk" if prediction_hard[0] == 1 else "✅ Low Default Risk"

    return result_hard, result_soft

# ------------- User Input Section -------------
print("\n🔍 Enter loan application details to predict default risk:")
features = [
    "Loan Amount", "Funded Amount", "Interest Rate (%)", "Annual Income", 
    "Debt-to-Income Ratio", "Delinquencies (2yrs)", "Credit Inquiries (6mths)", 
    "Open Credit Lines", "Revolving Balance", "Revolving Utilization (%)", "Total Credit Lines"
]

feature_columns = [
    'loan_amnt', 'funded_amnt', 'int_rate', 'annual_inc', 'dti',
    'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'revol_bal', 'revol_util', 'total_acc'
]

user_input = []
for feature in features:
    while True:
        try:
            value = float(input(f"Enter {feature}: "))
            user_input.append(value)
            break
        except ValueError:
            print("Please enter a valid number.")

# Convert input to DataFrame
user_df = pd.DataFrame([user_input], columns=feature_columns)
res1, res2 = predict_loan_default(user_df)
print(f"Hard voting prediction: {res1}")
print(f"Soft voting prediction: {res2}")
