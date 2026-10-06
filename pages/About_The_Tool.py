import streamlit as st

st.set_page_config(
    page_title="About the Tool",
    layout="wide"
)

st.markdown("""
<style>
    .card-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1.5rem;
        align-items: stretch;
    }
    @media (max-width: 700px) {
        .card-grid { grid-template-columns: 1fr; }
    }
    .card {
        border: 1px solid rgba(128, 128, 128, 0.3);
        border-radius: 12px;
        padding: 20px;
        background-color: var(--secondary-background-color);
        box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.08);
    }
    .card-title {
        font-size: 22px;
        font-weight: 600;
        color: var(--text-color);
    }
    .card-desc {
        font-size: 15px;
        color: var(--text-color);
        opacity: 0.7;
        margin: 8px 0 12px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("About NephroCheck")
st.caption("Understanding the purpose and limitations of this tool")
st.divider()

st.markdown("""
<div class="card-grid">
    <div class="card">
        <div class="card-title">Purpose</div>
        <div class="card-desc">NephroCheck uses common clinical parameters to estimate CKD risk using trained machine learning models, with additional interpretability through feature importance and SHAP analysis.</div>
    </div>
    <div class="card">
        <div class="card-title">Limitations</div>
        <div class="card-desc">Predictions are based on historical data and statistical models. The model is limited by the small-scale dataset used, so it is subject to inaccuracies. This tool is meant for demonstration only.</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.write("")

with st.expander("__How to Interpret Predictions__"):
    st.markdown("""
    - **Prediction Probability:** Represents the model’s estimated likelihood of CKD.
    - **Feature Importance:** Highlights which clinical indicators most influenced the result.
    - **SHAP Analysis:** Explains how individual features contributed to a specific prediction.
    """)

st.divider()

