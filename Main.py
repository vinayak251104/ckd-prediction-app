import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time 
import joblib 
# Set wide layout
st.set_page_config(page_title="KidneyScan – Predict CKD Using ML", layout="wide")
st.markdown("""
    <style>
        /* Smooth transition */
        html, body, .main, .block-container {
            transition: all 0.5s ease-in-out;
        }

        /* Match padding/margins */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }

        /* Avoid jumpy button/box rendering */
        .stButton>button, .stSelectbox, .stNumberInput {
            transition: all 0.3s ease-in-out;
        }
    </style>
""", unsafe_allow_html=True)

df=pd.read_csv('kidney_disease_dataset.csv')
# Custom CSS for outlined containers
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

# Centered Title
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.title('NephroCheck – Predict CKD Using ML')
    st.caption('__A Machine Learning Based App to Predict Chronic Kidney Disease (CKD)__')

st.divider()

# Two-column layout with full HTML block content inside
left_col, right_col = st.columns(2)
bottom_left_col, bottom_right_col = st.columns(2)
last_col=st.columns(1)
with left_col:
    st.markdown("""
        <div class="info-block">
            <div class="info-title">About the Tool</div>
            <p>This application uses trained machine learning models to predict Chronic Kidney Disease based on user input 
            features like GFR, BUN, Creatinine, Age, etc. It allows users to input data manually 
            and receive predictions instantly.</p>
        </div>
    """, unsafe_allow_html=True)

with right_col:
    st.markdown("""
        <div class="info-block">
            <div class="info-title">Team & Source</div>
            <p>Developed as a personal ML project using a real-world dataset sourced from Kaggle. 
            The goal is to demonstrate end-to-end machine learning deployment for healthcare data.</p>
            <p>Source: <a href="https://www.kaggle.com/datasets/miadul/kidney-disease-risk-dataset" target="_blank">Kaggle CKD Dataset</a></p>
        </div>
    """, unsafe_allow_html=True)
with bottom_left_col:
    with st.container():
        st.markdown("""
            <div class="info-block">
                <div class="info-title">Dataset Overview</div>
                <p>The dataset contains various features related to kidney function, including:</p>
        """, unsafe_allow_html=True)
        st.dataframe(df)

        # Close the div
        st.markdown("""</div>""", unsafe_allow_html=True)
with bottom_right_col:
    with st.container():
        st.markdown("""
            <div class="info-block">
                <div class="info-title">Using This Tool</div>
                <p>To predict your CKD status, simply enter the values from your test report in the form provided in Prediction Page.</p>
                <p>The model will analyze the data and return a prediction instantly. Please note that this is a <b>prediction tool </b>, not a medical diagnosis.</p>
                <p>Make sure the input values are accurate and recent to ensure meaningful results.</p>
            </div>
        """, unsafe_allow_html=True)
with bottom_right_col:
    st.markdown("""
        <div class="info-block">
            <div class="info-title">Key Predictive Features</div>
            <p>Our models highlight <b>GFR</b>, <b>BUN</b>, and <b>Creatinine Level</b> as the most impactful features in predicting CKD.</p>
            <p>These variables strongly correlate with kidney function and were consistently significant across different models, including <b>Random Forest</b> and <b>XGBoost</b>.</p>
        </div>
    """, unsafe_allow_html=True)









