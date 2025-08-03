import streamlit as st
from Main import centered_title
st.set_page_config(
    page_title="About the Tool",
    layout="wide"
)
st.markdown("""
<style>
    .card {
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 20px;
        background-color: #ffffff;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
    .card-title {
        font-size: 22px;
        font-weight: 600;
    }
    .card-desc {
        font-size: 15px;
        color: #555555;
        margin: 8px 0 12px 0;
    }
</style>
            """, unsafe_allow_html=True)

st.title("About NephroCheck")
st.caption("Understanding the purpose and limitations of this tool")
st.divider()

col1,col2= st.columns(2)

with col1:
    st.markdown("""
    <div class="card">
        <div class="card-title">Purpose</div>
        <div class="card-desc">
            NephroCheck uses common clinical parameters to estimate CKD risk using
        trained machine learning models, with additional interpretability through
        feature importance and SHAP analysis.
        </div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="card">
        <div class="card-title">Limitations</div>
        <div class="card-desc">
            Predictions are based on historical data and statistical models. The Model is limited based on the small scale dataset used, there is subject to inaccuracies. This tool is meant to be used for demonstration only.
        </div>
    </div>
    """, unsafe_allow_html=True)

with st.expander("__How to Interpret Predictions__"):
    st.markdown("""
    - **Prediction Probability:** Represents the model’s estimated likelihood of CKD.
    - **Feature Importance:** Highlights which clinical indicators most influenced the result.
    - **SHAP Analysis:** Explains how individual features contributed to a specific prediction.
    """)

st.divider()


