# 📡 Predicting Customer Churn

A machine learning project that predicts whether a customer will churn (cancel their subscription) using behavioral and demographic features. Built for the [Kaggle Playground Series S6E3](https://www.kaggle.com/competitions/playground-series-s6e3) competition and deployed as an interactive Streamlit web application.

---

## 🎯 Project Overview

Customer churn is one of the most critical challenges for subscription-based businesses. This project tackles the problem end-to-end:

- **Exploratory Data Analysis (EDA)** to understand customer behavior patterns
- **Feature Engineering** with risk profiles, tenure groups, and interaction variables
- **Machine Learning** with LightGBM for high-accuracy churn prediction
- **Interactive Web App** built with Streamlit for real-time predictions

---

## 📁 Project Structure

```
Predicting-Customer-Churn/
│
├── predict-customer-churn-with-ml.ipynb   # EDA, feature engineering & model training notebook
├── train_model.py                         # Script to train and serialize the model
├── app.py                                 # Streamlit web application
│
├── model.pkl                              # Trained LightGBM model
├── features.pkl                           # Feature list used by the model
├── thresholds.pkl                         # Threshold values for feature engineering
│
├── train.zip                              # Training dataset
├── test.zip                               # Test dataset
├── requirements.txt                       # Python dependencies
└── LICENSE
```

---

## ✨ Features

### 🔧 Feature Engineering
The model uses a rich set of engineered features derived from raw customer data:

- **Tenure groups** — Customers segmented into 5 groups by months of tenure
- **Risk profiles** — `HighRiskProfile`, `LowRiskProfile`, `HighRiskCombo` flags capturing combinations of high-risk behaviors
- **Risk scores** — `ContractRisk`, `PaymentRisk`, `InternetRisk` ordinal scores per category
- **Charge ratios** — `MonthlyToTotalRatio`, `ChargesPerMonth` to detect spending anomalies
- **Service count** — Total number of add-on services subscribed
- **Ordinal encoding** — For internet-related services (Online Security, Tech Support, etc.)

### 🌐 Streamlit App
The interactive web app allows users to input customer details across three panels:

- 👤 **Customer Info** — Gender, seniority, partner, dependents, tenure
- 📶 **Services** — Phone, internet, streaming, security, backup options
- 💳 **Contract & Payment** — Contract type, billing method, monthly and total charges

After clicking **Predict**, the app displays:
- ✅ / ⚠️ Churn prediction (Yes/No)
- Churn probability percentage
- Expandable table of all computed features

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/tugcesi/Predicting-Customer-Churn.git
cd Predicting-Customer-Churn
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python train_model.py
```

This generates `model.pkl`, `features.pkl`, and `thresholds.pkl`.

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

---

## 🧪 Model

| Detail | Value |
|---|---|
| Algorithm | LightGBM |
| Task | Binary Classification |
| Target | `Churn` (1 = churned, 0 = retained) |
| Evaluation Metric | AUC-ROC |
| Competition | Kaggle Playground Series S6E3 |

---

## 📦 Dependencies

```
streamlit
pandas
numpy
scikit-learn
lightgbm
joblib
```

---

## 📊 Dataset

The dataset is sourced from the [Kaggle Playground Series S6E3](https://www.kaggle.com/competitions/playground-series-s6e3) competition. It contains telecom customer data including demographic information, subscribed services, contract details, and billing information.

Download the data from Kaggle and place `train.csv` and `test.csv` (or the zipped versions) in the project root before running `train_model.py`.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
