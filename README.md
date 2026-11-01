# Classification Model Deployment with Streamlit

## a. Problem Statement
This project implements, evaluates, and compares five machine learning classifiers on a high-dimensional dataset to predict binary outcomes and deploys the model via an interactive Streamlit dashboard.

## b. Dataset Description
- **Instances:** 1,000 samples
- **Features:** 14 continuous numerical features
- **Target:** Binary classification (0 or 1)

## c. Github Repository Link
https://github.com/YOUR_GITHUB_USERNAME/ML-Assignment-2

## d. Models Used & Evaluation Metrics Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.8650 | 0.9223 | 0.8496 | 0.9057 | 0.8767 | 0.7297 |
| Decision Tree | 0.9000 | 0.8990 | 0.8981 | 0.9151 | 0.9065 | 0.7992 |
| KNN | 0.9550 | 0.9873 | 0.9533 | 0.9623 | 0.9577 | 0.9097 |
| Naive Bayes | 0.8350 | 0.9062 | 0.8349 | 0.8585 | 0.8465 | 0.6685 |
| Support Vector Machine | 0.9600 | 0.9924 | 0.9455 | 0.9811 | 0.9630 | 0.9203 |

## e. Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| Logistic Regression | Baseline linear model. Performs reasonably (86.5% accuracy) but misses non-linear decision patterns. |
| Decision Tree | Captures feature interaction effectively (90.0% accuracy), but shows moderate variance. |
| KNN | Performs strongly (95.5% accuracy) after feature scaling, capturing local clusters accurately. |
| Naive Bayes | Fast training speed, but conditional independence assumption degrades performance (83.5% accuracy). |
| Support Vector Machine | Optimal classifier with RBF kernel, achieving top performance across all metrics. |
| **Overall Winner** | **Support Vector Machine (SVM)** (Accuracy: 0.9600, AUC: 0.9924, MCC: 0.9203). |
```[cite: 3, 4]

---

### Step 6: Upload Project to GitHub

1. Go to [GitHub.com](https://github.com) and sign in.
2. Click **New Repository**.
3. Name it `ML-Assignment-2`, select **Public**, and click **Create repository**.
4. Upload all files into your repository so it matches this structure:
   ```text
   ML-Assignment-2/
   ├── app.py
   ├── requirements.txt
   ├── README.md
   ├── test_data.csv
   ├── train_and_save.py
   └── model/
       ├── scaler.pkl
       ├── logistic_regression.pkl
       ├── decision_tree.pkl
       ├── knn.pkl
       ├── naive_bayes.pkl
       └── svm.pkl
   ```[cite: 3]

---

### Step 7: Deploy App on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) (or `streamlit.io/cloud`)[cite: 4].
2. Sign in with your GitHub account[cite: 4].
3. Click **New app**[cite: 4].
4. Select your repository (`ML-Assignment-2`), branch (`main`), and set Main file path to `app.py`[cite: 4].
5. Click **Deploy!**[cite: 4]
6. Copy your live web app URL (e.g., `[https://your-app-name.streamlit.app](https://your-app-name.streamlit.app)`)[cite: 1, 4].

---

### Step 8: Create & Submit PDF Document

Open Word/Google Docs, paste the following content in this **exact order**, and export as a single **PDF file**[cite: 1, 2]:

1. **GitHub Repository Link:**[cite: 1]
   `[https://github.com/YOUR_GITHUB_USERNAME/ML-Assignment-2](https://github.com/YOUR_GITHUB_USERNAME/ML-Assignment-2)`
2. **Live Streamlit App Link:**[cite: 1, 2]
   `[https://your-app-name.streamlit.app](https://your-app-name.streamlit.app)`
3. **Screenshot of Execution on BITS Virtual Lab:**[cite: 2, 3]
   *(Insert your `bits_lab_screenshot.png` image taken in Step 2)*
4. **README Content:**[cite: 2, 3]
   *(Copy and paste the entire Markdown text from Step 5 into the document)*

Save as `Assignment2_ML.pdf` and submit it on the WILP portal before **08-Nov-2026 23:59 PM**[cite: 1, 6].