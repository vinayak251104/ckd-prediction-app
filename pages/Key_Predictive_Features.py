import streamlit as st

st.set_page_config(page_title="Key Predictive Features", layout="wide")

st.markdown("""
<style>
    .card-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.5rem;
        align-items: stretch;
    }
    @media (max-width: 900px) {
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
    .card ul {
        margin-bottom: 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("Key Predictive Features")
st.caption("Why the model focuses on these clinical indicators")
st.divider()

st.markdown("""
The model consistently relies on a small set of clinically meaningful features.
These variables directly reflect kidney function and metabolic stress.
""")

st.divider()

st.markdown("""
<div class="card-grid">
    <div class="card">
        <div class="card-title">GFR</div>
        <div class="card-desc">Primary indicator of kidney function</div>
        <ul>
            <li>Lower GFR → reduced filtration capacity</li>
            <li>Strong inverse relationship with CKD risk</li>
            <li>Early clinical warning sign</li>
        </ul>
    </div>
    <div class="card">
        <div class="card-title">BUN</div>
        <div class="card-desc">Marker of waste accumulation</div>
        <ul>
            <li>Elevated when kidneys fail to clear urea</li>
            <li>Commonly high in CKD patients</li>
            <li>Reflects metabolic imbalance</li>
        </ul>
    </div>
    <div class="card">
        <div class="card-title">Creatinine</div>
        <div class="card-desc">Indicator of filtration efficiency</div>
        <ul>
            <li>Waste product filtered by kidneys</li>
            <li>Higher levels suggest impaired clearance</li>
            <li>Used alongside GFR in diagnosis</li>
        </ul>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

st.info(
    "These features remained influential across multiple models, reinforcing "
    "their clinical relevance rather than model-specific behavior."
)
