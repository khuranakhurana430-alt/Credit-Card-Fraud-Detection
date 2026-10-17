import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# 1. PAGE CONFIG
st.set_page_config(
    page_title="FraudGuard AI", 
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .big-font { font-size:20px !important; }
    .metric-card { background-color: #1E1E1E; padding: 15px; border-radius: 10px; border: 1px solid #333; }
    </style>
""", unsafe_allow_html=True)

# Load artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load('fraud_model.pkl')
    scaler = joblib.load('scaler.pkl')
    threshold = joblib.load('threshold.pkl')
    return model, scaler, threshold

model, scaler, threshold = load_artifacts()

# 2. SIDEBAR
logo_url = "https://cdn-icons-png.flaticon.com/512/2886/2886670.png"
st.sidebar.image(logo_url, width=100)
st.sidebar.title("FraudGuard AI")
st.sidebar.markdown("---")

st.sidebar.info(
    "**How to use this app:**\n\n"
    "1. Go to the **Fraud Simulator** tab.\n"
    "2. Use the preset buttons at the top to load a test transaction.\n"
    "3. Or, manually adjust the **V1-V5 sliders** (the most impactful features).\n"
    "4. Click **Analyze Transaction** to see the prediction."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Information")
st.sidebar.write(f"**Accuracy:** 99.95%")
st.sidebar.write(f"**Precision:** 91.8%")
st.sidebar.write(f"**Decision Threshold:** {threshold:.2f}")

st.sidebar.markdown("---")
st.sidebar.markdown("[🔗 View Source Code on GitHub](https://github.com/khuranakhurana430-alt/Credit-Card-Fraud-Detection)")

# 3. MAIN PAGE HEADER
st.title("🛡️ FraudGuard AI")
st.markdown("An advanced Machine Learning system designed to detect fraudulent credit card transactions in real-time using XGBoost.")

# Create Tabs
tab1, tab2 = st.tabs(["💳 Fraud Simulator", "📊 Model Insights"])

# --- TAB 1: SIMULATOR ---
with tab1:
    st.subheader("Transaction Simulator")
    
    # Session state to handle presets
    if 'v1' not in st.session_state: st.session_state.v1 = 0.0
    if 'v2' not in st.session_state: st.session_state.v2 = 0.0
    if 'v3' not in st.session_state: st.session_state.v3 = 0.0
    if 'v4' not in st.session_state: st.session_state.v4 = 0.0
    if 'v5' not in st.session_state: st.session_state.v5 = 0.0
    if 'amount' not in st.session_state: st.session_state.amount = 100.0

    # Preset Buttons
    st.write("**Quick Test Scenarios:**")
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        if st.button("🟢 Load Legitimate Transaction"):
            st.session_state.v1, st.session_state.v2, st.session_state.v3 = 0.0, 0.0, 0.0
            st.session_state.v4, st.session_state.v5 = 0.0, 0.0
            st.session_state.amount = 45.00
            st.rerun()
            
    with col_btn2:
        if st.button("🚨 Load Suspicious Transaction"):
            st.session_state.v1, st.session_state.v2, st.session_state.v3 = -4.5, 1.2, -5.1
            st.session_state.v4, st.session_state.v5 = 3.4, -2.5
            st.session_state.amount = 1500.00
            st.rerun()

    st.markdown("---")

    # Input Form
    with st.form("fraud_form"):
        col1, col2 = st.columns(2)
        with col1:
            amount = st.number_input("Transaction Amount ($)", min_value=0.0, max_value=25000.0, value=st.session_state.amount, step=10.0)
        with col2:
            time = st.number_input("Time (Seconds)", min_value=0.0, max_value=172800.0, value=50000.0, step=1000.0)
            
        st.markdown("**Top PCA Features (V1-V5)**")
        st.caption("Adjust these sliders to change the transaction profile. Extreme values trigger fraud alerts.")
        
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: v1 = st.slider("V1", -10.0, 10.0, st.session_state.v1)
        with c2: v2 = st.slider("V2", -10.0, 10.0, st.session_state.v2)
        with c3: v3 = st.slider("V3", -10.0, 10.0, st.session_state.v3)
        with c4: v4 = st.slider("V4", -10.0, 10.0, st.session_state.v4)
        with c5: v5 = st.slider("V5", -10.0, 10.0, st.session_state.v5)

        submit_button = st.form_submit_button("🔍 Analyze Transaction", type="primary", use_container_width=True)

    if submit_button:
        # Prepare Data
        input_data = {'scaled_amount': amount, 'scaled_time': time}
        input_data.update({f'V{i}': 0.0 for i in range(6, 29)}) # Fill V6-V28 with 0
        input_data.update({'V1': v1, 'V2': v2, 'V3': v3, 'V4': v4, 'V5': v5})
        
        input_df = pd.DataFrame([input_data])
        
        # Scale
        input_df['scaled_amount'] = scaler.transform(input_df['scaled_amount'].values.reshape(-1, 1))
        input_df['scaled_time'] = scaler.transform(input_df['scaled_time'].values.reshape(-1, 1))

        # Predict
        proba = model.predict_proba(input_df)[0, 1]
        prediction = "FRAUD DETECTED" if proba >= threshold else "LEGITIMATE"

        st.divider()
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            st.subheader("Analysis Result")
            if proba >= threshold:
                st.error(f"### 🚨 {prediction}")
                st.write("This transaction exhibits patterns consistent with **fraud**. It should be blocked or flagged for manual review.")
            else:
                st.success(f"### ✅ {prediction}")
                st.write("This transaction appears to be **legitimate**. No immediate action required.")
            
            # Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = proba * 100,
                number = {'suffix': "%"},
                title = {'text': "Fraud Probability", 'font': {'size': 24}},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "rgba(0,0,0,0)"},
                    'steps': [
                        {'range': [0, threshold * 100], 'color': "lightgreen"},
                        {'range': [threshold * 100, 100], 'color': "salmon"}
                    ],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': proba * 100}
                }
            ))
            fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with res_col2:
            st.subheader("Transaction Breakdown")
            st.metric(label="Transaction Amount", value=f"${amount:,.2f}")
            st.metric(label="Model Confidence", value=f"{max(proba, 1-proba)*100:.2f}%")
            
            if proba >= threshold:
                st.warning("⚠️ **Action Required:** Freeze card and contact customer.")
            else:
                st.info("ℹ️ Transaction approved and processed normally.")


# --- TAB 2: INSIGHTS ---
with tab2:
    st.subheader("Model Insights & Feature Importance")
    st.write("Because the dataset is highly imbalanced (99.8% legitimate, 0.2% fraud), the model relies heavily on specific PCA components to catch anomalies.")
    
    col_chart, col_text = st.columns([1, 1])
    
    with col_chart:
        try:
            importances = model.feature_importances_
            feature_names = ['scaled_amount', 'scaled_time'] + [f'V{i}' for i in range(1, 29)]
            
            fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            fi_df = fi_df.sort_values(by='Importance', ascending=True).tail(10)
            
            fig_fi = px.bar(
                fi_df, x='Importance', y='Feature', orientation='h',
                color='Importance', color_continuous_scale='Reds',
                title="Top 10 Most Important Features"
            )
            fig_fi.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_fi, use_container_width=True)
        except Exception as e:
            st.error(f"Could not load chart: {e}")
            
    with col_text:
        st.markdown("### Understanding the Data")
        st.write("""
        * **V1 - V28:** These are the result of a **PCA (Principal Component Analysis)** transformation. 
        * To protect user privacy, the original features (like location, merchant type, etc.) were anonymized by the bank.
        * **V3, V4, V11, V12, and V14** are typically the strongest indicators of fraudulent behavior in this specific dataset.
        * **Amount & Time:** These were scaled using a StandardScaler to match the PCA components.
        
        ### Model Performance
        * **XGBoost** was chosen for its ability to handle highly imbalanced data.
        * **SMOTE** (Synthetic Minority Over-sampling Technique) was applied during training to generate synthetic fraud examples, allowing the model to learn the patterns of rare events.
        * **Threshold Tuning:** Instead of a default 0.5 threshold, the threshold was optimized to **{:.2f}** to balance Precision and Recall.
        """.format(threshold))