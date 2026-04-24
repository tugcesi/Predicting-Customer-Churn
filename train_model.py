"""
train_model.py
--------------
Müşteri churn tahmin modelini eğitir ve kaydeder.

Kullanım:
  1. train.zip dosyasını proje dizinine koy (veya train.csv olarak çıkar).
  2. `python train_model.py` komutunu çalıştır.
  3. model.pkl, features.pkl ve thresholds.pkl dosyaları oluşturulur.
"""

import zipfile
import io
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from lightgbm import LGBMClassifier

# ─── 1. VERİYİ YÜKLEYELİM ───────────────────────────────────────────────────
print("Veri yükleniyor...")
try:
    # train.zip varsa doğrudan oku
    with zipfile.ZipFile("train.zip") as z:
        csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
        with z.open(csv_name) as f:
            df = pd.read_csv(f)
    print(f"  train.zip içinden '{csv_name}' okundu. Satır: {len(df)}")
except FileNotFoundError:
    # Yoksa train.csv'yi dene
    df = pd.read_csv("train.csv")
    print(f"  train.csv okundu. Satır: {len(df)}")

# ─── 2. FEATURE ENGINEERING ──────────────────────────────────────────────────
print("Feature engineering uygulanıyor...")

# Hedef değişken
df["Churn"] = df["Churn"].replace({"Yes": 1, "No": 0})

# Binary özellikler
df["gender"] = df["gender"].replace({"Male": 1, "Female": 0}).astype(int)
df[["Partner", "Dependents", "PhoneService", "PaperlessBilling"]] = (
    df[["Partner", "Dependents", "PhoneService", "PaperlessBilling"]]
    .replace({"Yes": 1, "No": 0})
    .astype(int)
)

# Tenure özellikleri
def categorize_tenure(tenure):
    if tenure < 6:
        return 1
    elif tenure < 12:
        return 2
    elif tenure < 24:
        return 3
    elif tenure < 48:
        return 4
    else:
        return 5

df["TenureGroup"] = df["tenure"].apply(categorize_tenure)
df["IsNewCustomer"] = (df["tenure"] < 6).astype(int)
df["IsLongTermCustomer"] = (df["tenure"] > 48).astype(int)

# Ücret özellikleri
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
df["ChargesPerMonth"] = df["TotalCharges"] / (df["tenure"] + 1)
df["MonthlyToTotalRatio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1)

monthly_q75 = float(df["MonthlyCharges"].quantile(0.75))
monthly_median = float(df["MonthlyCharges"].median())
df["IsHighMonthlyCharges"] = (df["MonthlyCharges"] > monthly_q75).astype(int)

# İnternet servisi özellikleri
df["HasInternet"] = (df["InternetService"] != "No").astype(int)
df["IsFiberOptic"] = (df["InternetService"] == "Fiber optic").astype(int)

# Servis sayısı
service_cols = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
df["TotalServices"] = (df[service_cols] == "Yes").sum(axis=1)

# Sözleşme özellikleri
df["IsLongTermContract"] = df["Contract"].isin(["One year", "Two year"]).astype(int)
df["IsMonthlyContract"] = (df["Contract"] == "Month-to-month").astype(int)

# Ödeme özellikleri
df["HasAutomaticPayment"] = df["PaymentMethod"].isin(
    ["Credit card (automatic)", "Bank transfer (automatic)"]
).astype(int)

# Risk profilleri
df["HighRiskProfile"] = (
    (df["IsMonthlyContract"] == 1)
    & (df["IsNewCustomer"] == 1)
    & (df["IsHighMonthlyCharges"] == 1)
).astype(int)

df["LowRiskProfile"] = (
    (df["IsLongTermContract"] == 1)
    & (df["IsLongTermCustomer"] == 1)
    & (df["HasAutomaticPayment"] == 1)
    & (df["TotalServices"] >= 3)
).astype(int)

df["HighRiskCombo"] = (
    (df["IsFiberOptic"] == 1)
    & (df["IsLongTermContract"] == 0)
    & (df["MonthlyCharges"] > monthly_median)
).astype(int)

# Risk skorları
contract_risk = {"Month-to-month": 3, "One year": 2, "Two year": 1}
df["ContractRisk"] = df["Contract"].map(contract_risk)

internet_risk = {"No": 1, "DSL": 2, "Fiber optic": 3}
df["InternetServiceRisk"] = df["InternetService"].map(internet_risk)

payment_risk = {
    "Electronic check": 3,
    "Mailed check": 2,
    "Credit card (automatic)": 1,
    "Bank transfer (automatic)": 1,
}
df["PaymentRisk"] = df["PaymentMethod"].map(payment_risk)

# Ordinal encoding
for col in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]:
    df[f"{col}_Ordinal"] = df[col].map({"No": 0, "Yes": 1, "No internet service": 2})
    df[f"{col}_HasService"] = (df[col] == "Yes").astype(int)
    df[f"{col}_NoInternet"] = (df[col] == "No internet service").astype(int)

# ─── 3. ÖZELLİK SEÇİMİ ───────────────────────────────────────────────────────
FEATURES = [
    "HighRiskCombo",
    "ContractRisk",
    "PaymentRisk",
    "TenureGroup",
    "IsFiberOptic",
    "OnlineSecurity_Ordinal",
    "TechSupport_Ordinal",
    "OnlineBackup_Ordinal",
    "DeviceProtection_Ordinal",
    "MonthlyToTotalRatio",
    "IsLongTermCustomer",
    "HasAutomaticPayment",
    "IsNewCustomer",
    "PaperlessBilling",
    "MonthlyCharges",
    "Dependents",
    "SeniorCitizen",
    "Partner",
]

X = df[FEATURES]
y = df["Churn"].astype(int)

print(f"  Özellik sayısı: {len(FEATURES)}, Örnek sayısı: {len(X)}")

# ─── 4. MODELİ EĞİTELİM ──────────────────────────────────────────────────────
print("Model eğitiliyor (LGBMClassifier)...")

model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=40,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
print(f"  Cross-validation ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Tüm veri ile eğit
model.fit(X, y)
print("  Model eğitimi tamamlandı.")

# ─── 5. KAYDET ───────────────────────────────────────────────────────────────
joblib.dump(model, "model.pkl")
joblib.dump(FEATURES, "features.pkl")

# Eşik değerlerini kaydet (uygulama için gerekli)
thresholds = {
    "monthly_charges_q75": monthly_q75,
    "monthly_charges_median": monthly_median,
}
joblib.dump(thresholds, "thresholds.pkl")

print("Dosyalar kaydedildi: model.pkl, features.pkl, thresholds.pkl")
print("Eğitim tamamlandı!")
