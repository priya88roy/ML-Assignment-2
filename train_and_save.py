import os
import joblib
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef
)

# 1. Create dataset (14 features, 1000 instances to meet assignment specs)
X, y = make_classification(n_samples=1000, n_features=14, n_informative=10, n_redundant=4, random_state=42)
feature_names = [f'feature_{i+1}' for i in range(14)]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# 2. Split dataset into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=['target']), df['target'], test_size=0.2, random_state=42
)

# Save test dataset (features + target column) for Streamlit upload
test_df = X_test.copy()
test_df['target'] = y_test
test_df.to_csv('test_data.csv', index=False)
print("Saved test_data.csv successfully!")

# 3. Fit scaler for models requiring normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

os.makedirs('model', exist_ok=True)
joblib.dump(scaler, 'model/scaler.pkl')

# 4. Initialize the 5 required classifiers
models = {
    'logistic_regression': LogisticRegression(),
    'decision_tree': DecisionTreeClassifier(random_state=42),
    'knn': KNeighborsClassifier(),
    'naive_bayes': GaussianNB(),
    'svm': SVC(probability=True, random_state=42)
}

# 5. Train, evaluate, and save each model
results = []
scaled_models = ['logistic_regression', 'knn', 'svm']

for name, model in models.items():
    if name in scaled_models:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate required evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    results.append({
        'ML Model Name': name.replace('_', ' ').title(),
        'Accuracy': round(acc, 4),
        'AUC': round(auc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1': round(f1, 4),
        'MCC': round(mcc, 4)
    })
    
    # Save model file into model/ directory
    joblib.dump(model, f'model/{name}.pkl')

print("\n--- MODEL COMPARISON TABLE ---")
metrics_df = pd.DataFrame(results)
print(metrics_df.to_string(index=False))