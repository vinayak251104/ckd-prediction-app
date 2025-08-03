import streamlit as st
import pandas as pd

st.set_page_config(page_title="Dataset Overview", layout="wide")

st.title("Dataset Overview")
st.divider()

st.markdown("""
The dataset used in this project contains anonymized clinical measurements
related to kidney function and associated health conditions.
""")

st.subheader("Dataset Source")
st.markdown("""
- Source: Kaggle – Kidney Disease Risk Dataset  
- Data includes numerical and categorical features commonly used in
  CKD assessment.
""")

df = pd.read_csv("kidney_disease_dataset.csv")

st.subheader("Dataset Summary")
st.markdown(f"""
- **Total Records:** {df.shape[0]}  
- **Total Features:** {df.shape[1]}
""")

st.subheader("Feature Overview")
st.markdown("""
Key features include:
- Glomerular Filtration Rate (GFR)
- Blood Urea Nitrogen (BUN)
- Creatinine Level
- Urine Output
- Age
- Diabetes, Hypertension, Dialysis status
""")

with st.expander("Preview Dataset"):
    st.dataframe(df.head())
