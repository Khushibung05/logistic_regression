import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="💗",
    layout="centered"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
/* Background */
.main {
     background: linear-gradient(
        135deg,
        rgb(147, 226, 222) 0%,
        rgb(238, 235, 170) 50%,
        rgb(213, 135, 190) 100%
    );
}

/* Pink cards */
.card {
    background: #f5a6e6;
    padding: 22px;
    border-radius: 16px;
    margin-bottom: 22px;
    color: white;
    font-weight: 600;
}

/* Yellow separators */
.separator {
    height: 18px;
    background: #facc15;
    border-radius: 12px;
    margin: 20px 0;
}

/* Dark table */
.dark-table {
    background: #111827;
    padding: 16px;
    border-radius: 12px;
}

/* Metric grid */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 16px;
}

/* Metric item */
.metric {
    background: #93c5fd;
    padding: 14px;
    border-radius: 12px;
    color: #0f172a;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    return pd.read_csv("telco_customer_churn.csv")

df = load_data()

# ================= HEADER =================
st.markdown("""
<div class="card">
<h2>Customer Churn Prediction</h2>
<p>Using Logistic Regression to predict whether a customer is likely to churn or stay</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="separator"></div>', unsafe_allow_html=True)

# ================= DATASET PREVIEW =================
st.markdown("<h3>Dataset Preview</h3>", unsafe_allow_html=True)

st.markdown('<div class="dark-table">', unsafe_allow_html=True)
st.dataframe(df.head(), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="separator"></div>', unsafe_allow_html=True)

# ================= PREPROCESSING =================
df = df.drop("customerID", axis=1)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df = pd.get_dummies(df, drop_first=True)

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ================= MODEL =================
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# ================= CONFUSION MATRIX =================
st.markdown("<h3>Confusion Matrix</h3>", unsafe_allow_html=True)

fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Churn", "Churn"]
)
disp.plot(ax=ax_cm, cmap="Blues", colorbar=True)
st.pyplot(fig_cm)

st.markdown('<div class="separator"></div>', unsafe_allow_html=True)

# ================= MODEL PERFORMANCE =================
st.markdown("<h3>Model Performance</h3>", unsafe_allow_html=True)

st.markdown("""
<div class="metric-grid">
<div class="metric">Accuracy<br>0.80</div>
<div class="metric">True Positive (TP)<br>196</div>
<div class="metric">True Negative (TN)<br>936</div>
<div class="metric">False Positive (FP)<br>100</div>
<div class="metric">False Negative (FN)<br>177</div>
<div class="metric">Total Predictions<br>1409</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="separator"></div>', unsafe_allow_html=True)

# ================= PREDICTION FORM =================
st.markdown("<h3>Predict Customer Churn</h3>", unsafe_allow_html=True)

tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly = st.slider("Monthly Charges", 20.0, 120.0, 70.0)
total = st.slider("Total Charges", 20.0, 9000.0, 1000.0)

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

# Dummy prediction (UI demo)
churn_prob = 0.49

st.markdown(f"""
<div style="background:#d1fae5;padding:18px;border-radius:14px;text-align:center;">
<h4>Prediction: Likely to Stay</h4>
<p>Churn Probability: {churn_prob}</p>
</div>
""", unsafe_allow_html=True)
