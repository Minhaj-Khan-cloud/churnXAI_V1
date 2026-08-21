import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# পেজ কনফিগারেশন
st.set_page_config(page_title="ChurnXAI | Retention Dashboard", layout="wide", page_icon="📊")

# মডিউল ইমপোর্ট
try:
    from src.data_prep import load_and_clean_data, preprocess_and_balance_data
    from src.shap_engine import load_explainer, generate_waterfall_plot
    from src.database_action import init_db, save_high_risk_customer, generate_retention_offer, create_pdf_report
except Exception as e:
    st.error(f"❌ Module Import Error: {e}. Make sure 'src' folder exists and has __init__.py")
    st.stop()

# ==========================================
# 1. Data & Model Loading (Cached for Speed)
# ==========================================
@st.cache_resource
def load_project_assets():
    init_db()  # ডাটাবেজ ইনিশিয়ালাইজ করা
    
    raw_df = load_and_clean_data("dataset/telco_churn.csv")
    raw_df = raw_df.reset_index(drop=True)  # ইনডেক্স মিসম্যাচ ফিক্স
    
    X_bal, y_bal, cols = preprocess_and_balance_data(raw_df)
    
    model_path = "models/best_churn_model.pkl"
    if not os.path.exists(model_path):
        return None, None, None, None
        
    model = joblib.load(model_path)
    explainer = load_explainer(model_path, X_bal)
    
    return raw_df, X_bal, model, explainer

# এসেট লোড করা
raw_data, X_processed, model, explainer = load_project_assets()

if model is None:
    st.warning("⚠️ Model not found! Please run `src/model_training.py` first.")
    st.stop()

# ==========================================
# 2. UI Layout & Dashboard
# ==========================================
st.title("📊 ChurnXAI: AI-Powered Customer Retention")
st.markdown("A Machine Learning System to Predict & Prevent Customer Churn using XAI.")
st.markdown("---")

# সাইডবার: টিম মেম্বারদের তথ্য
st.sidebar.header("👨‍💻 Project Team Roles")
st.sidebar.info(
    "**Member 1:** Data Engineering & SMOTE\n\n"
    "**Member 2:** ML Engine (Random Forest)\n\n"
    "**Member 3:** SHAP Explainable AI\n\n"
    "**Member 4:** Auto Action Engine & DB"
)

# মূল স্ক্রিন দুই ভাগে বিভক্ত
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("👤 Select Customer Profile")
    st.markdown("Search or pick any customer from the database.")
    
    if 'customerID' in raw_data.columns:
        selected_cust_id = st.selectbox("Choose Customer ID:", raw_data['customerID'].unique())
        single_customer_raw = raw_data[raw_data['customerID'] == selected_cust_id]
        customer_index = single_customer_raw.index[0]
    else:
        customer_index = st.selectbox("Choose Customer Index:", range(len(raw_data)))
        single_customer_raw = raw_data.iloc[[customer_index]]

    single_customer_processed = X_processed.iloc[[customer_index]]
    
    st.write("**Customer Quick Info:**")
    st.dataframe(single_customer_raw[['gender', 'Contract', 'MonthlyCharges', 'TotalCharges']].T.astype(str))

    # প্রেডিক্ট বাটন
    if st.button("Predict Churn Risk 🚀", use_container_width=True, type="primary"):
        risk_prob = model.predict_proba(single_customer_processed)[0][1]
        risk_percentage = round(risk_prob * 100, 2)
        
        st.session_state['risk'] = risk_percentage
        st.session_state['customer_data'] = single_customer_processed
        st.session_state['customer_id'] = customer_index
        st.session_state['raw_customer'] = single_customer_raw

with col2:
    if 'risk' in st.session_state:
        risk = st.session_state['risk']
        customer_data = st.session_state['customer_data']
        customer_id = st.session_state['customer_id']
        raw_cust = st.session_state['raw_customer']
        
        # 1. রিস্ক স্কোর দেখানো
        st.subheader("🎯 Prediction Result")
        if risk > 50:
            st.error(f"🔥 **High Churn Risk Detected:** The probability of this customer leaving is **{risk}%**")
        else:
            st.success(f"✅ **Customer is Safe:** The probability of leaving is only **{risk}%**")
            
        # 2. SHAP Explanation
        st.markdown("---")
        st.subheader("🧠 Explainable AI: Why is the model saying this?")
        st.markdown("Red bars push the risk higher, Blue bars push the risk lower.")
        
        fig = generate_waterfall_plot(explainer, customer_data)
        if fig:
            st.pyplot(fig)
            
        # 3. Action Engine & PDF Download
        if risk > 50:
            st.markdown("---")
            st.subheader("⚡ Automated Retention Action (Triggered)")
            
            with st.spinner("Saving to SQLite database & generating official PDF report..."):
                cust_name_val = raw_cust['customerID'].values[0] if 'customerID' in raw_cust.columns else f"Customer_{customer_id}"
                save_high_risk_customer(cust_name_val, risk, "Identified via SHAP Analysis")
                
                offers_list = generate_retention_offer(raw_cust)
                pdf_bytes = create_pdf_report(cust_name_val, raw_cust, risk, offers_list)
                
                st.success("📁 High-risk profile permanently logged in SQLite database.")
                
                display_text = f"### 📋 Recommended Retention Strategies\n\n"
                for i, off in enumerate(offers_list, 1):
                    display_text += f"{i}. {off}\n\n"
                st.info(display_text)
                
                st.download_button(
                    label="📥 Download Official PDF Report (.pdf)",
                    data=pdf_bytes,
                    file_name=f"Retention_Report_{cust_name_val}.pdf",
                    mime="application/pdf"
                )