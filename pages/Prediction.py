import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import shap
import joblib
import plotly.graph_objects as go

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Prediction Page",
    layout="wide",
    initial_sidebar_state="expanded"
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

# ---------------- Sidebar: Prediction Threshold ----------------
# This slider controls how strict the model is when deciding CKD vs Non-CKD
# Higher threshold → stricter decision
# Lower threshold → more sensitive decision
with st.sidebar:
    st.markdown("### Prediction Settings")
    threshold = st.slider(
        "CKD risk threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05
    )



def get_value(key, fallback):
    return st.session_state[key] if st.session_state[key] is not None else fallback

@st.cache_resource
def load_model():
    return joblib.load('final_xg_model_for_ckd_status.pkl')

model = load_model()

# ---------------- Styling ----------------
st.markdown("""
    <style>
        .centered-subheader {
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 15px;
            margin-left: 40px;
            margin-right: 35px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
        .info-block {
            border: 1px solid #d3d3d3;
            border-radius: 10px;
            padding: 20px;
            background-color: #ffffff;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
            margin-top: 10px;
        }
        .info-title {
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

df = pd.read_csv('kidney_disease_dataset.csv')

# ---------------- Title ----------------
st.title('Chronic Kidney Disease Status Prediction')

st.caption("The CKD Risk threshold controls how strict the model is when classifying CKD."
    " Higher values make the model more conservative, while lower values increase sensitivity.")


st.divider()
# ---------------- Default Session Values ----------------
features = {
    'Age': 0,
    'Creatinine_Level': 0.0,
    'BUN': 0.0,
    'Diabetes': 'No',
    'Hypertension': 'No',
    'GFR': 0.0,
    'Urine_Output': 0.0,
    'Dialysis_Needed': 'No'
}

for key, val in features.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ---------------- Input Form ----------------
with st.form(key='Input Form'):
    st.subheader('Risk Factor Input Form')

    age = st.number_input(
        'Age',
        min_value=0,
        max_value=int(df['Age'].max()),
        step=1,
        value=get_value('Age', 0)
    )

    creatinine_level = st.number_input(
        'Creatinine Level (mg/dL)',
        min_value=0.0,
        max_value=float(df['Creatinine_Level'].max()),
        step=0.1,
        value=get_value('Creatinine_Level', 0.0)
    )

    bun = st.number_input(
        'BUN (mg/dL)',
        min_value=0.0,
        max_value=float(df['BUN'].max()),
        step=0.1,
        value=get_value('BUN', 0.0)
    )

    diabetes = st.selectbox(
        'Diabetes',
        options=['Yes', 'No'],
        index=0 if get_value('Diabetes', 'No') == 'Yes' else 1
    )

    hypertension = st.selectbox(
        'Hypertension',
        options=['Yes', 'No'],
        index=0 if get_value('Hypertension', 'No') == 'Yes' else 1
    )

    gfr = st.number_input(
        'GFR (mL/min/1.73 m²)',
        min_value=0.0,
        max_value=float(df['GFR'].max()),
        step=0.1,
        value=get_value('GFR', 0.0)
    )

    urine_output = st.number_input(
        'Urine Output (mL/day)',
        min_value=0.0,
        max_value=float(df['Urine_Output'].max()),
        step=0.1,
        value=get_value('Urine_Output', 0.0)
    )

    dialysis_needed = st.selectbox(
        'Dialysis Needed',
        options=['Yes', 'No'],
        index=0 if get_value('Dialysis_Needed', 'No') == 'Yes' else 1
    )

    submit_button = st.form_submit_button("Submit")

    if submit_button:
        st.session_state['Age'] = age
        st.session_state['Creatinine_Level'] = creatinine_level
        st.session_state['BUN'] = bun
        st.session_state['Diabetes'] = diabetes
        st.session_state['Hypertension'] = hypertension
        st.session_state['GFR'] = gfr
        st.session_state['Urine_Output'] = urine_output
        st.session_state['Dialysis_Needed'] = dialysis_needed

    # ---------------- Validation ----------------
    errors = []
    if age <= 0: errors.append("Age must be greater than 0.")
    if creatinine_level <= 0: errors.append("Creatinine Level must be greater than 0.")
    if bun <= 0: errors.append("BUN must be greater than 0.")
    if gfr <= 0: errors.append("GFR must be greater than 0.")
    if urine_output <= 0: errors.append("Urine Output must be greater than 0.")

    if errors:
        st.error("Please correct the following:")
        for err in errors:
            st.markdown(f"<span style='color:red'>• {err}</span>", unsafe_allow_html=True)
    else:
        st.success("Form submitted successfully!")

        features = {
            'Age': age,
            'Creatinine_Level': creatinine_level,
            'BUN': bun,
            'Diabetes': 1 if diabetes == 'Yes' else 0,
            'Hypertension': 1 if hypertension == 'Yes' else 0,
            'GFR': gfr,
            'Urine_Output': urine_output,
            'Dialysis_Needed': 1 if dialysis_needed == 'Yes' else 0
        }

        st.divider()

        # ---------------- Display User Input ----------------
        col4, col5, col6 = st.columns([1, 3, 1])
        with col5:
            st.markdown('<div class="centered-subheader">User Information</div>', unsafe_allow_html=True)
            for key, value in features.items():
                st.write(f"Your {key} is: {value}")

        # ---------------- Model Prediction ----------------
        # Model outputs probability of CKD
        pred_proba = model.predict_proba(pd.DataFrame([features]))[0][1]

        # Use user-defined threshold instead of hardcoded 0.5
        pred_label = 1 if pred_proba >= threshold else 0

        display_proba = pred_proba * 100 if pred_label == 1 else (1 - pred_proba) * 100
        result_text = "positive (CKD detected)" if pred_label == 1 else "negative (No CKD)"

        # ---------------- Animated Gauge ----------------
        placeholder = st.empty()
        for i in range(0, int(round(pred_proba * 100)) + 1, 2):
            fig = go.Figure(go.Indicator(
                mode="gauge",
                value=i,
                title={'text': "CKD Probability (%)"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': 'lightgreen'},
                        {'range': [30, 70], 'color': 'khaki'},
                        {'range': [70, 100], 'color': 'salmon'}
                    ]
                }
            ))
            placeholder.plotly_chart(fig, use_container_width=True)
            time.sleep(0.015)

        # ---------------- Result ----------------
        st.markdown(
            f"""<div class="centered-subheader">
            You have a {round(display_proba, 2)}% chance of being {result_text}.
            </div>""",
            unsafe_allow_html=True
        )

        # Explain threshold transparently
        st.caption(
            f"Decision threshold set to {threshold:.2f} "
            f"(Prediction is CKD if probability ≥ threshold)"
        )

        st.markdown("""
            <div class="info-block">
                <div class="info-title">Interpretation & Next Steps</div>
                <p>If the prediction is <b>positive</b>, please consult a nephrologist or general physician for professional evaluation.</p>
                <p>If the prediction is <b>negative</b>, that does not guarantee you are risk-free. Always cross-check with actual lab reports.</p>
                <p>This tool is intended for informational use only, not a substitute for clinical diagnosis.</p>
            </div>
        """, unsafe_allow_html=True)



