"""
app.py
------
Müşteri churn tahmini için Streamlit uygulaması.

Kullanım:
  streamlit run app.py

Not: Çalıştırmadan önce `python train_model.py` komutu ile
     model.pkl, features.pkl ve thresholds.pkl dosyalarını oluşturun.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ─── SAYFA AYARLARI ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Müşteri Churn Tahmini",
    page_icon="📡",
    layout="wide",
)

# ─── MODEL YÜKLEMESİ ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not (
        os.path.exists("model.pkl")
        and os.path.exists("features.pkl")
        and os.path.exists("thresholds.pkl")
    ):
        return None, None, None
    model = joblib.load("model.pkl")
    features = joblib.load("features.pkl")
    thresholds = joblib.load("thresholds.pkl")
    return model, features, thresholds


model, features, thresholds = load_model()

# ─── BAŞLIK ──────────────────────────────────────────────────────────────────
st.title("📡 Müşteri Churn Tahmin Uygulaması")
st.markdown(
    """
Bu uygulama, bir müşterinin **churn etme (aboneliğini iptal etme)** ihtimalini
makine öğrenmesi ile tahmin eder.  
Aşağıdaki formu doldurup **"Tahmin Et"** butonuna tıklayın.
"""
)

if model is None:
    st.error(
        "⚠️ Model dosyaları bulunamadı! "
        "Lütfen önce `python train_model.py` komutunu çalıştırarak "
        "model.pkl, features.pkl ve thresholds.pkl dosyalarını oluşturun."
    )
    st.stop()

st.divider()

# ─── GİRDİ FORMU ─────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Müşteri Bilgileri")
    gender = st.selectbox("Cinsiyet", ["Male", "Female"])
    senior_citizen = st.selectbox("Yaşlı Vatandaş mı?", ["No", "Yes"])
    partner = st.selectbox("Eşi var mı?", ["No", "Yes"])
    dependents = st.selectbox("Bakmakla yükümlü kişi var mı?", ["No", "Yes"])
    tenure = st.slider("Müşteri Süresi (ay)", min_value=0, max_value=72, value=12)

with col2:
    st.subheader("📶 Hizmet Bilgileri")
    phone_service = st.selectbox("Telefon Hizmeti", ["No", "Yes"])
    multiple_lines = st.selectbox(
        "Birden Fazla Hat", ["No", "Yes", "No phone service"]
    )
    internet_service = st.selectbox("İnternet Hizmeti", ["DSL", "Fiber optic", "No"])

    internet_opts = ["No", "Yes", "No internet service"]
    online_security = st.selectbox("Çevrimiçi Güvenlik", internet_opts)
    online_backup = st.selectbox("Çevrimiçi Yedekleme", internet_opts)
    device_protection = st.selectbox("Cihaz Koruma", internet_opts)
    tech_support = st.selectbox("Teknik Destek", internet_opts)
    streaming_tv = st.selectbox("TV Yayın Servisi", internet_opts)
    streaming_movies = st.selectbox("Film Yayın Servisi", internet_opts)

with col3:
    st.subheader("💳 Sözleşme & Ödeme")
    contract = st.selectbox(
        "Sözleşme Tipi", ["Month-to-month", "One year", "Two year"]
    )
    paperless_billing = st.selectbox("Kağıtsız Fatura", ["No", "Yes"])
    payment_method = st.selectbox(
        "Ödeme Yöntemi",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
    monthly_charges = st.number_input(
        "Aylık Ücret ($)", min_value=0.0, max_value=200.0, value=65.0, step=0.5
    )
    total_charges = st.number_input(
        "Toplam Ücret ($)", min_value=0.0, max_value=10000.0, value=500.0, step=1.0
    )

st.divider()

# ─── TAHMİN FONKSİYONU ───────────────────────────────────────────────────────
def build_features(
    gender, senior_citizen, partner, dependents, tenure,
    phone_service, multiple_lines, internet_service,
    online_security, online_backup, device_protection,
    tech_support, streaming_tv, streaming_movies,
    contract, paperless_billing, payment_method,
    monthly_charges, total_charges,
    thresholds,
):
    """Kullanıcı girdilerinden model özelliklerini üretir."""

    monthly_q75 = thresholds["monthly_charges_q75"]
    monthly_median = thresholds["monthly_charges_median"]

    # Binary dönüşümler
    gender_enc = 1 if gender == "Male" else 0
    senior_enc = 1 if senior_citizen == "Yes" else 0
    partner_enc = 1 if partner == "Yes" else 0
    dependents_enc = 1 if dependents == "Yes" else 0
    paperless_enc = 1 if paperless_billing == "Yes" else 0

    # Tenure özellikleri
    if tenure < 6:
        tenure_group = 1
    elif tenure < 12:
        tenure_group = 2
    elif tenure < 24:
        tenure_group = 3
    elif tenure < 48:
        tenure_group = 4
    else:
        tenure_group = 5

    is_new_customer = 1 if tenure < 6 else 0
    is_long_term_customer = 1 if tenure > 48 else 0

    # Ücret özellikleri
    charges_per_month = total_charges / (tenure + 1)
    monthly_to_total_ratio = monthly_charges / (total_charges + 1)
    is_high_monthly = 1 if monthly_charges > monthly_q75 else 0

    # İnternet servisi
    has_internet = 1 if internet_service != "No" else 0
    is_fiber_optic = 1 if internet_service == "Fiber optic" else 0

    # Toplam servis sayısı
    service_vals = [
        online_security, online_backup, device_protection,
        tech_support, streaming_tv, streaming_movies,
    ]
    total_services = sum(1 for v in service_vals if v == "Yes")

    # Sözleşme
    is_long_term_contract = 1 if contract in ["One year", "Two year"] else 0
    is_monthly_contract = 1 if contract == "Month-to-month" else 0

    # Ödeme
    has_automatic_payment = 1 if payment_method in [
        "Credit card (automatic)", "Bank transfer (automatic)"
    ] else 0

    # Risk profilleri
    high_risk_profile = int(
        is_monthly_contract == 1 and is_new_customer == 1 and is_high_monthly == 1
    )
    low_risk_profile = int(
        is_long_term_contract == 1
        and is_long_term_customer == 1
        and has_automatic_payment == 1
        and total_services >= 3
    )
    high_risk_combo = int(
        is_fiber_optic == 1
        and is_long_term_contract == 0
        and monthly_charges > monthly_median
    )

    # Risk skorları
    contract_risk = {"Month-to-month": 3, "One year": 2, "Two year": 1}[contract]
    internet_risk = {"No": 1, "DSL": 2, "Fiber optic": 3}[internet_service]
    payment_risk = {
        "Electronic check": 3,
        "Mailed check": 2,
        "Credit card (automatic)": 1,
        "Bank transfer (automatic)": 1,
    }[payment_method]

    # Ordinal encoding
    ordinal_map = {"No": 0, "Yes": 1, "No internet service": 2}
    online_security_ord = ordinal_map.get(online_security, 0)
    tech_support_ord = ordinal_map.get(tech_support, 0)
    online_backup_ord = ordinal_map.get(online_backup, 0)
    device_protection_ord = ordinal_map.get(device_protection, 0)

    row = {
        "HighRiskCombo": high_risk_combo,
        "ContractRisk": contract_risk,
        "PaymentRisk": payment_risk,
        "TenureGroup": tenure_group,
        "IsFiberOptic": is_fiber_optic,
        "OnlineSecurity_Ordinal": online_security_ord,
        "TechSupport_Ordinal": tech_support_ord,
        "OnlineBackup_Ordinal": online_backup_ord,
        "DeviceProtection_Ordinal": device_protection_ord,
        "MonthlyToTotalRatio": monthly_to_total_ratio,
        "IsLongTermCustomer": is_long_term_customer,
        "HasAutomaticPayment": has_automatic_payment,
        "IsNewCustomer": is_new_customer,
        "PaperlessBilling": paperless_enc,
        "MonthlyCharges": monthly_charges,
        "Dependents": dependents_enc,
        "SeniorCitizen": senior_enc,
        "Partner": partner_enc,
    }
    return pd.DataFrame([row])


# ─── TAHMİN BUTONU ───────────────────────────────────────────────────────────
if st.button("🔍 Tahmin Et", type="primary", use_container_width=True):
    input_df = build_features(
        gender, senior_citizen, partner, dependents, tenure,
        phone_service, multiple_lines, internet_service,
        online_security, online_backup, device_protection,
        tech_support, streaming_tv, streaming_movies,
        contract, paperless_billing, payment_method,
        monthly_charges, total_charges,
        thresholds,
    )

    # Özellik sırasını modelle eşleştir
    input_df = input_df[features]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.divider()
    res_col1, res_col2 = st.columns([2, 1])

    with res_col1:
        if prediction == 1:
            st.error(
                f"## ⚠️ Müşteri Churn Edebilir!\n\n"
                f"Bu müşterinin aboneliğini iptal etme olasılığı yüksek."
            )
        else:
            st.success(
                f"## ✅ Müşteri Churn Etmez\n\n"
                f"Bu müşterinin aboneliğini sürdürmesi bekleniyor."
            )

    with res_col2:
        st.metric(
            label="Churn Olasılığı",
            value=f"{probability * 100:.1f}%",
        )

    # Detay tablosu
    with st.expander("📊 Hesaplanan Özellikler"):
        st.dataframe(input_df.T.rename(columns={0: "Değer"}), use_container_width=True)
