import streamlit as st

st.set_page_config(
    page_title="...",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* Remove underline from all links */
    a {
        text-decoration: none !important;
    }

    /* Optional: nicer hover behavior */
    a:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

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

def centered_title(text,caption):
    col4, col5, col6 = st.columns([1, 3, 1])
    with col5:
        st.title(text)
        st.caption(caption)

centered_title('NephroCheck - Predict CKD Using ML','A Machine Learning Based App to Predict Chronic Kidney Disease Risk')
st.divider()

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)


with col1:
    st.markdown("""
    <div class="card">
        <div class="card-title">About The Tool</div>
        <div class="card-desc">
            Understand the purpose and limitations of NephroCheck.
        </div>
        <a href="About_The_Tool">Learn more →</a>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <div class="card-title">Dataset Overview</div>
        <div class="card-desc">
            Learn about the dataset, features used and data source.
        </div>
        <a href="Dataset_Overview">Explore dataset →</a>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <div class="card-title">Key Predictive Features</div>
        <div class="card-desc">
            High-level intuition behind the most influential indicators.
        </div>
        <a href="Key_Predictive_Features">View features →</a>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="card">
        <div class="card-title">Model Analysis</div>
        <div class="card-desc">
            Feature distributions, SHAP explanations, and performance metrics.
        </div>
        <a href="Analysis">Go to analysis →</a>
    </div>
    """, unsafe_allow_html=True)

st.divider()

st.subheader("Make a Prediction")
st.markdown(
    '<a href="Prediction">Start prediction →</a>',
    unsafe_allow_html=True
)

st.divider()
st.caption(
    "⚠️ This application is for demonstration purposes only. "
    "It is not a medical diagnostic tool."
)


