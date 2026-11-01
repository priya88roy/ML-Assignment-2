import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)

st.set_page_config(page_title="ML Classifier Evaluation App", layout="wide")

st.title("Classification Model Evaluation Dashboard")
st.write("Upload test data, select a model, and review real-time classification metrics.")

# Sidebar Features
st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader("Upload test data (CSV)", type=["csv"])

model_options = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Support Vector Machine": "model/svm.pkl"
}

selected_model_name = st.sidebar.selectbox("Select ML Model", list(model_options.keys()))

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    
    if 'target' not in test_df.columns:
        st.error("Uploaded CSV must contain a 'target' column.")
    else:
        X_test = test_df.drop(columns=['target'])
        y_test = test_df['target']
        
        # Load saved model and scaler without retraining
        model = joblib.load(model_options[selected_model_name])
        scaler = joblib.load('model/scaler.pkl')
        
        # Scale features if required
        if selected_model_name in ["Logistic Regression", "KNN", "Support Vector Machine"]:
            X_eval = scaler.transform(X_test)
        else:
            X_eval = X_test
        
        # Make predictions
        y_pred = model.predict(X_eval)
        
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_eval)[:, 1]
        else:
            y_proba = y_pred
            
        # Calculate Metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        # Display Metrics
        st.subheader(f"Evaluation Metrics: {selected_model_name}")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Accuracy", f"{acc:.4f}")
        c2.metric("AUC", f"{auc:.4f}")
        c3.metric("Precision", f"{prec:.4f}")
        c4.metric("Recall", f"{rec:.4f}")
        c5.metric("F1 Score", f"{f1:.4f}")
        c6.metric("MCC Score", f"{mcc:.4f}")
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            st.pyplot(fig)
            
        with col2:
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.highlight_max(axis=0))
else:
    st.info("Upload `test_data.csv` in the sidebar to view evaluation results.")