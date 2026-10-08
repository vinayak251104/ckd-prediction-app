import time

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Prediction Page",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Styling (theme-aware) ----------------
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
    .info-block {
        border: 1px solid rgba(128, 128, 128, 0.3);
        border-radius: 12px;
        padding: 20px;
        background-color: var(--secondary-background-color);
        box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.08);
        margin-top: 10px;
    }
    .info-block p:last-child {
        margin-bottom: 0;
    }
    .info-title {
        font-size: 24px;
        font-weight: 600;
        margin-bottom: 10px;
        color: var(--text-color);
    }
</style>
""", unsafe_allow_html=True)


# ---------------- Loaders ----------------
@st.cache_resource
def load_model():
    return joblib.load('final_xg_model_for_ckd_status.pkl')


@st.cache_data
def load_data():
    return pd.read_csv('kidney_disease_dataset.csv')


model = load_model()
df = load_data()
MODEL_COLUMNS = list(model.feature_names_in_)
LAB_FEATURES = ['Creatinine_Level', 'BUN', 'GFR', 'Urine_Output']


# ---------------- Sidebar: Prediction Settings ----------------
with st.sidebar:
    st.markdown("### Prediction Settings")
    threshold = st.slider(
        "CKD risk threshold",
        min_value=0.1, max_value=0.9, value=0.5, step=0.05,
        help="The result is CKD if the estimated probability is at or above this value."
    )
    measurement_error = st.slider(
        "Assumed lab measurement error (%)",
        min_value=0, max_value=10, value=5, step=1,
        help="Lab values are never exact. The probability is estimated by re-running the "
             "model on many slightly perturbed copies of your inputs. Set to 0 to use the "
             "raw model output."
    ) / 100


def ckd_probability(row_df, err, n_draws=300):
    """Probability of CKD.

    err == 0  -> raw model probability.
    err  > 0  -> share of n_draws noisy copies (multiplicative lab error on
                 creatinine, BUN, GFR, urine output) that the model labels CKD.
    A fixed seed keeps the result stable when only the threshold slider moves.
    """
    row_df = row_df[MODEL_COLUMNS]
    if err == 0:
        return float(model.predict_proba(row_df)[0][1])
    rng = np.random.default_rng(0)
    batch = pd.concat([row_df] * n_draws, ignore_index=True).astype(float)
    noise = rng.normal(0, err, (n_draws, len(LAB_FEATURES)))
    batch[LAB_FEATURES] = (batch[LAB_FEATURES] * (1 + noise)).clip(lower=0)
    return float((model.predict(batch) == 1).mean())


def make_gauge(value, threshold_pct):
    return go.Figure(go.Indicator(
        mode="gauge",
        value=value,
        title={'text': "CKD Probability (%)"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': 'lightgreen'},
                {'range': [30, 70], 'color': 'khaki'},
                {'range': [70, 100], 'color': 'salmon'}
            ],
            'threshold': {'line': {'color': "black", 'width': 3},
                          'value': threshold_pct}
        }
    ))


# ---------------- Title ----------------
st.title('Chronic Kidney Disease Status Prediction')
st.caption(
    "The CKD risk threshold controls how strict the model is when classifying CKD. "
    "Higher values make the model more conservative, while lower values increase sensitivity."
)
st.divider()

# ---------------- Default Session Values ----------------
# Saved in session_state so entered values survive navigating to another page.
defaults = {
    'Age': 0,
    'Creatinine_Level': 0.0,
    'BUN': 0.0,
    'Diabetes': 'No',
    'Hypertension': 'No',
    'GFR': 0.0,
    'Urine_Output': 0.0,
    'Dialysis_Needed': 'No',
    'submitted': False,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ---------------- Input Form ----------------
with st.form(key='Input Form'):
    st.subheader('Risk Factor Input Form')

    age = st.number_input(
        'Age', min_value=0, max_value=int(df['Age'].max()), step=1,
        value=st.session_state['Age'])
    creatinine_level = st.number_input(
        'Creatinine Level (mg/dL)', min_value=0.0,
        max_value=float(df['Creatinine_Level'].max()), step=0.1,
        value=st.session_state['Creatinine_Level'])
    bun = st.number_input(
        'BUN (mg/dL)', min_value=0.0, max_value=float(df['BUN'].max()), step=0.1,
        value=st.session_state['BUN'])
    diabetes = st.selectbox(
        'Diabetes', options=['Yes', 'No'],
        index=0 if st.session_state['Diabetes'] == 'Yes' else 1)
    hypertension = st.selectbox(
        'Hypertension', options=['Yes', 'No'],
        index=0 if st.session_state['Hypertension'] == 'Yes' else 1)
    gfr = st.number_input(
        'GFR (mL/min/1.73 m²)', min_value=0.0, max_value=float(df['GFR'].max()),
        step=0.1, value=st.session_state['GFR'])
    urine_output = st.number_input(
        'Urine Output (mL/day)', min_value=0.0,
        max_value=float(df['Urine_Output'].max()), step=0.1,
        value=st.session_state['Urine_Output'])
    dialysis_needed = st.selectbox(
        'Dialysis Needed', options=['Yes', 'No'],
        index=0 if st.session_state['Dialysis_Needed'] == 'Yes' else 1)

    submit_button = st.form_submit_button("Submit")

if submit_button:
    st.session_state.update({
        'submitted': True,
        'Age': age,
        'Creatinine_Level': creatinine_level,
        'BUN': bun,
        'Diabetes': diabetes,
        'Hypertension': hypertension,
        'GFR': gfr,
        'Urine_Output': urine_output,
        'Dialysis_Needed': dialysis_needed,
    })

# ---------------- Results (only after the form has been submitted) ----------------
if st.session_state['submitted']:
    errors = []
    if age <= 0: errors.append("Age must be greater than 0.")
    if creatinine_level <= 0: errors.append("Creatinine Level must be greater than 0.")
    if bun <= 0: errors.append("BUN must be greater than 0.")
    if gfr <= 0: errors.append("GFR must be greater than 0.")
    if urine_output <= 0: errors.append("Urine Output must be greater than 0.")

    if errors:
        st.error("Please correct the following:\n\n" + "\n".join(f"- {e}" for e in errors))
    else:
        if submit_button:
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
        _, col5, _ = st.columns([1, 3, 1])
        with col5:
            st.markdown('<div class="centered-subheader">User Information</div>',
                        unsafe_allow_html=True)
            shown = {**features, 'Diabetes': diabetes, 'Hypertension': hypertension,
                     'Dialysis_Needed': dialysis_needed}
            for key, value in shown.items():
                st.write(f"Your {key} is: {value}")

        # ---------------- Prediction ----------------
        pred_proba = ckd_probability(pd.DataFrame([features]), measurement_error)
        pred_label = 1 if pred_proba >= threshold else 0
        result_text = "positive (CKD detected)" if pred_label == 1 else "negative (No CKD)"

        # ---------------- Gauge (animate only right after Submit) ----------------
        placeholder = st.empty()
        if submit_button:
            for i in range(0, int(round(pred_proba * 100)) + 1, 2):
                placeholder.plotly_chart(make_gauge(i, threshold * 100),
                                         use_container_width=True)
                time.sleep(0.015)
        placeholder.plotly_chart(make_gauge(pred_proba * 100, threshold * 100),
                                 use_container_width=True)

        # ---------------- Result ----------------
        st.markdown(
            f"""<div class="centered-subheader">
            Estimated CKD probability: {pred_proba * 100:.1f}%<br>
            Prediction at threshold {threshold:.2f}: {result_text}
            </div>""",
            unsafe_allow_html=True
        )
        if measurement_error > 0:
            st.caption(
                f"Probability accounts for an assumed ±{measurement_error * 100:.0f}% lab "
                f"measurement error, so borderline values give intermediate probabilities. "
                f"The black line on the gauge marks your threshold ({threshold:.2f}). "
                f"Set the error to 0% in the sidebar to see the raw model output."
            )
        else:
            st.caption(
                f"Raw model output. Decision threshold: {threshold:.2f} "
                f"(prediction is CKD if probability ≥ threshold). This model's raw "
                f"probabilities are almost always near 0% or 100%, so the threshold "
                f"rarely changes the result."
            )

        st.markdown("""
            <div class="info-block">
                <div class="info-title">Interpretation & Next Steps</div>
                <p>If the prediction is <b>positive</b>, please consult a nephrologist or general physician for professional evaluation.</p>
                <p>If the prediction is <b>negative</b>, that does not guarantee you are risk-free. Always cross-check with actual lab reports.</p>
                <p>This tool is intended for informational use only, not a substitute for clinical diagnosis.</p>
            </div>
        """, unsafe_allow_html=True)


