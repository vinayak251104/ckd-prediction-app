import streamlit as st

st.set_page_config(page_title="Key Predictive Features", layout="wide")


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


st.title("Key Predictive Features")
st.caption("Why the model focuses on these clinical indicators")
st.divider()

# --- High-level context (short, not a paragraph) ---
st.markdown("""
The model consistently relies on a small set of clinically meaningful features.
These variables directly reflect kidney function and metabolic stress.
""")

st.divider()

# --- Feature cards layout ---
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
        <div class="card-title">GFR</div>
        <div class="card-desc">Primary indicator of kidney function</div>
        <ul class="card-list">
            <li>Lower GFR → reduced filtration capacity</li>
            <li>Strong inverse relationship with CKD risk</li>
            <li>Early clinical warning sign</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <div class="card-title">BUN</div>
        <div class="card-desc">Marker of waste accumulation</div>
        <ul class="card-list">
            <li>Elevated when kidneys fail to clear urea</li>
            <li>Commonly high in CKD patients</li>
            <li>Reflects metabolic imbalance</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <div class="card-title">Creatinine</div>
        <div class="card-desc">Indicator of filtration efficiency</div>
        <ul class="card-list">
            <li>Waste product filtered by kidneys</li>
            <li>Higher levels suggest impaired clearance</li>
            <li>Used alongside GFR in diagnosis</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


st.divider()

# --- Closing insight ---
st.info(
    "These features remained influential across multiple models, reinforcing "
    "their clinical relevance rather than model-specific behavior."
)
