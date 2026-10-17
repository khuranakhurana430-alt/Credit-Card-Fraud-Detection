# 🛡️ FraudGuard AI

**Real-time Credit Card Fraud Detection & Analytics Dashboard**

FraudGuard AI is a machine learning-powered web application designed to detect fraudulent credit card transactions in real-time. It uses an optimized XGBoost model trained on highly imbalanced data, combined with a beautiful, interactive Streamlit dashboard.

---

## 🚀 Features
- **Real-time Prediction:** Instantly predicts if a transaction is fraudulent or legitimate.
- **Interactive Dashboard:** Beautiful dark "Grey Ambient" UI with live charts.
- **Advanced Machine Learning:** Handles extreme class imbalance using SMOTE and XGBoost.
- **Feature Importance:** Visualizes which transaction features the model pays attention to.
- **Random Data Generator:** One-click button to generate test data for quick demos.

## 🛠️ Tech Stack
- **Frontend:** Streamlit, Plotly
- **Backend/ML:** Python, XGBoost, Scikit-learn, Imbalanced-learn
- **Data Processing:** Pandas, NumPy
- **Model Saving:** Joblib

## 📊 Model Performance
Because fraud datasets are highly imbalanced (only 0.172% fraud), we focus on the F1-Score and Recall rather than just Accuracy.

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 99.9% |
| **Precision** | 92.1% |
| **Recall** | 88.5% |
| **F1-Score** | 90.2% |
| **ROC-AUC** | 98.5% |

## 📂 Project Structure
```text
Credit_Card_Fraud_PROJECT/
├── app.py                # Streamlit dashboard code
├── train.ipynb           # Model training notebook
├── creditcard.csv        # Dataset (Ignored in Git due to size)
├── fraud_model.pkl       # Trained XGBoost model
├── scaler.pkl            # StandardScaler for Amount & Time
├── threshold.pkl         # Optimal prediction threshold
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
