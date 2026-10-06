import streamlit as st

st.set_page_config(
    page_title="NephroCheck",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* Remove underline from all links */
    a {
        text-decoration: none !important;
    }
    a:hover {
        text-decoration: underline !important;
    }

    /* Card grid: cards in a row always match height */
    .card-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1.5rem;
        align-items: stretch;
    }
    @media (max-width: 700px) {
        .card-grid { grid-template-columns: 1fr; }
    }

    /* Theme-aware cards */
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


def centered_title(text, caption):
    _, col, _ = st.columns([1, 3, 1])
    with col:
        st.title(text)
        st.caption(caption)


centered_title(
    "NephroCheck - Predict CKD Using ML",
    "A Machine Learning Based App to Predict Chronic Kidney Disease Risk",
)
st.divider()

st.markdown("""
<div class="card-grid">
    <div class="card">
        <div class="card-title">About The Tool</div>
        <div class="card-desc">Understand the purpose and limitations of NephroCheck.</div>
        <a href="About_The_Tool">Learn more →</a>
    </div>
    <div class="card">
        <div class="card-title">Dataset Overview</div>
        <div class="card-desc">Learn about the dataset, features used and data source.</div>
        <a href="Dataset_Overview">Explore dataset →</a>
    </div>
    <div class="card">
        <div class="card-title">Key Predictive Features</div>
        <div class="card-desc">High-level intuition behind the most influential indicators.</div>
        <a href="Key_Predictive_Features">View features →</a>
    </div>
    <div class="card">
        <div class="card-title">Model Analysis</div>
        <div class="card-desc">Feature distributions, SHAP explanations, and performance metrics.</div>
        <a href="Analysis">Go to analysis →</a>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

st.subheader("Make a Prediction")
st.markdown('<a href="Prediction">Start prediction →</a>', unsafe_allow_html=True)

st.divider()
st.caption(
    "⚠️ This application is for demonstration purposes only. "
    "It is not a medical diagnostic tool."
)

