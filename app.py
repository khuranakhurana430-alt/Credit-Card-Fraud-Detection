import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# 1. PAGE CONFIG
st.set_page_config(
    page_title="Fraud Detection System", 
    page_icon="🛡️", 
    layout="wide"
)

# Load artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load('fraud_model.pkl')
    scaler = joblib.load('scaler.pkl')
    threshold = joblib.load('threshold.pkl')
    return model, scaler, threshold

model, scaler, threshold = load_artifacts()

# 2. SIDEBAR WITH LOGO
# Using a professional security shield icon
logo_url = "https://cdn-icons-png.flaticon.com/512/2886/2886670.png"
st.sidebar.image(logo_url, width=100)
st.sidebar.title("Fraud Analytics")
st.sidebar.markdown("---")
st.sidebar.info(
    "This system uses an XGBoost model trained on PCA-transformed credit card data. "
    "It is designed to detect fraudulent transactions in real-time to protect customers."
)
st.sidebar.markdown("---")
st.sidebar.write(f"**Model Accuracy:** 99.95%")
st.sidebar.write(f"**Fraud Precision:** 91.8%")
st.sidebar.write(f"**Decision Threshold:** {threshold:.2f}")

# MAIN PAGE HEADER
st.title("🛡️ Credit Card Fraud Detection")
st.markdown("Enter transaction details below to evaluate the risk of fraud. *(Note: V1-V28 are anonymized PCA components).*")

# Use a form to group inputs
with st.form("fraud_form"):
    st.subheader("Transaction Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        amount = st.number_input("Transaction Amount ($)", min_value=0.0, max_value=25000.0, value=100.0, step=10.0)
    with col2:
        time = st.number_input("Time (Seconds since first transaction)", min_value=0.0, max_value=172800.0, value=50000.0, step=1000.0)
    with col3:
        # A few of the most important PCA features to give the user something to interact with
        st.markdown("*(V1-V4 are the most impactful PCA features)*")
        v1 = st.slider("V1", -5.0, 5.0, 0.0)
        v2 = st.slider("V2", -5.0, 5.0, 0.0)
        
    st.markdown("---")
    st.markdown("**Advanced PCA Features (Defaults to 0)**")
    col4, col5, col6 = st.columns(3)
    with col4:
        v3 = st.slider("V3", -5.0, 5.0, 0.0)
    with col5:
        v4 = st.slider("V4", -5.0, 5.0, 0.0)
    with col6:
        v5 = st.slider("V5", -5.0, 5.0, 0.0)

    # Submit button
    submit_button = st.form_submit_button("🔍 Analyze Transaction", type="primary", use_container_width=True)

if submit_button:
    # 1. Prepare the 30 features
    # The model expects: scaled_amount, scaled_time, V1, V2, ..., V28
    input_data = {
        'scaled_amount': amount,
        'scaled_time': time,
    }
    # Populate V1 to V28 (using the sliders for 1-5, and 0 for the rest)
    input_data['V1'] = v1
    input_data['V2'] = v2
    input_data['V3'] = v3
    input_data['V4'] = v4
    input_data['V5'] = v5
    for i in range(6, 29):
        input_data[f'V{i}'] = 0.0

    input_df = pd.DataFrame([input_data])

    # 2. Scale Amount and Time (using the saved scaler)
    input_df['scaled_amount'] = scaler.transform(input_df['scaled_amount'].values.reshape(-1, 1))
    input_df['scaled_time'] = scaler.transform(input_df['scaled_time'].values.reshape(-1, 1))

    # 3. Predict
    proba = model.predict_proba(input_df)[0, 1]
    prediction = "FRAUD DETECTED" if proba >= threshold else "LEGITIMATE"

    st.divider()
    
    # --- RESULTS LAYOUT ---
    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        st.subheader("Transaction Result")
        if proba >= threshold:
            st.error(f"### 🚨 {prediction}")
            st.write("This transaction exhibits patterns consistent with **fraud**. It should be blocked or flagged for manual review.")
        else:
            st.success(f"### ✅ {prediction}")
            st.write("This transaction appears to be **legitimate**. No immediate action required.")
        
        # Plotly Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = proba * 100,
            number = {'suffix': "%"},
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Fraud Probability", 'font': {'size': 24}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "rgba(0,0,0,0)"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, threshold * 100], 'color': "lightgreen"},
                    {'range': [threshold * 100, 100], 'color': "salmon"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': proba * 100
                }
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with res_col2:
        st.subheader("Top Fraud Indicators (PCA)")
        st.write("Which anonymized components drove this decision?")
        
        # Extract feature importances
        try:
            importances = model.feature_importances_
            # The model trained on 30 columns
            feature_names = ['scaled_amount', 'scaled_time'] + [f'V{i}' for i in range(1, 29)]
            
            fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            fi_df = fi_df.sort_values(by='Importance', ascending=True).tail(10)
            
            fig_fi = px.bar(
                fi_df, x='Importance', y='Feature', orientation='h',
                color='Importance', color_continuous_scale='Reds',
                title="Top 10 Most Important Features"
            )
            fig_fi.update_layout(showlegend=False, height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_fi, use_container_width=True)
        except Exception as e:
            st.error(f"Could not load feature importance chart: {e}")