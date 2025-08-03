import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time 
import shap 
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
df=pd.read_csv('kidney_disease_dataset.csv')
y=df['CKD_Status']
X=df.drop(columns=['CKD_Status'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
model=joblib.load('final_xg_model_for_ckd_status.pkl')
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 100

st.set_page_config(
    page_title="CKD Data Insights & Feature Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)
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


model=joblib.load('final_xg_model_for_ckd_status.pkl')
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
        .info-subtitle {
            font-size: 18px;
            font-weight: 400;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
        .centered-subheader {
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 5px;
            margin-top: 20px;
            margin-left: 40px;
        }
    </style>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
        .centered-subheader_matrix {
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 5px;
            margin-top: 20px;
            margin-left: 80px;
            margin-right: 70px;
        }
    </style>
""", unsafe_allow_html=True)

st.title('CKD Data Insights & Feature Analysis')
st.caption('Exploring feature distributions, importance, and model performance')
st.divider()

st.markdown("""
<div class="info-block">
    <div class="info-title">How to Read This Page</div>
    <p>
        This page explores how the dataset behaves, which features drive the
        model’s predictions, and how the trained model performs on held-out data.
        Expand each section to view detailed visualizations.
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()
st.markdown(
    """
    <div class="info-block">
        <div class="info-title">Understanding Feature Distributions</div>
        <p>Visualizing the distribution of key features helps identify patterns and anomalies in the dataset. 
        This analysis is crucial for understanding how different variables relate to CKD status.</p>
    </div>
    """, unsafe_allow_html=True
)

with st.expander("__Feature Distributions Overview__"):
    col1,col2,col3,col4=st.columns(4)
    col5,col6,col7,col8=st.columns(4)
    with col1:
        st.markdown('<div class="centered-subheader">GFR</div>', unsafe_allow_html=True)
        plt.figure(figsize=(5,3))
        gfr_data = pd.read_csv('kidney_disease_dataset.csv')['GFR']
        sns.histplot(gfr_data, kde=True, color='blue')
        st.pyplot(plt.gcf())
        plt.clf()
    with col2:
        st.markdown('<div class="centered-subheader">BUN</div>', unsafe_allow_html=True)
        plt.figure(figsize=(5,3))
        bun_data = pd.read_csv('kidney_disease_dataset.csv')['BUN']
        sns.histplot(bun_data, kde=True, color='blue')
        st.pyplot(plt.gcf())
        plt.clf()
    with col3:
        st.markdown('<div class="centered-subheader">Urine</div>', unsafe_allow_html=True)
        plt.figure(figsize=(5,3))
        urine_data = pd.read_csv('kidney_disease_dataset.csv')['Urine_Output']
        sns.histplot(urine_data, kde=True, color='blue')
        st.pyplot(plt.gcf())
        plt.clf()
    with col4:
        st.markdown('<div class="centered-subheader">Age</div>', unsafe_allow_html=True)
        plt.figure(figsize=(5,3))
        age_data = pd.read_csv('kidney_disease_dataset.csv')['Age']
        sns.histplot(age_data, kde=True, color='blue')
        st.pyplot(plt.gcf())
        plt.clf()
    with col5:
        st.markdown('<div class="centered-subheader">Creatinine</div>', unsafe_allow_html=True)
        plt.figure(figsize=(5,3))
        creatinine_data = pd.read_csv('kidney_disease_dataset.csv')['Creatinine_Level']
        sns.histplot(creatinine_data, kde=True, color='blue')
        st.pyplot(plt.gcf())
        plt.clf()
    with col6:
        st.markdown('<div class="centered-subheader">Hypertension</div>', unsafe_allow_html=True)
        plt.figure(figsize=(5,3))
        hyper_data = pd.read_csv('kidney_disease_dataset.csv')['Hypertension']
        sns.histplot(hyper_data, kde=True, color='blue')
        st.pyplot(plt.gcf(), use_container_width=True)
        plt.clf()
    with col7:
        st.markdown('<div class="centered-subheader">Diabetes</div>', unsafe_allow_html=True)
        plt.figure(figsize=(5,3))
        diabetes_data = pd.read_csv('kidney_disease_dataset.csv')['Diabetes']
        sns.histplot(diabetes_data, kde=True, color='blue')
        st.pyplot(plt.gcf())
        plt.clf()
    with col8:
        st.markdown('<div class="centered-subheader">Dialysis</div>', unsafe_allow_html=True)
        plt.figure(figsize=(5,3))
        dialysis_data = pd.read_csv('kidney_disease_dataset.csv')['Dialysis_Needed']
        sns.histplot(dialysis_data, kde=True, color='blue')
        st.pyplot(plt.gcf())
        plt.clf()



st.divider()

st.markdown("""
    <div class="info-block">
        <div class="info-title">Feature Importance Analysis</div>
        <p>Understanding which features most influence CKD predictions helps in model interpretability and trust.</p>
    </div>
""", unsafe_allow_html=True)

with st.expander("__Feature Importance Overview__"):
    col9, col10 = st.columns(2)
    with col9:
        st.markdown('<div class="centered-subheader">Feature Correlation Heatmap</div>', unsafe_allow_html=True)
        df = pd.read_csv("kidney_disease_dataset.csv")
        plt.figure(figsize=(10,5))
        sns.heatmap(df.corr(), annot=True, cmap="viridis", fmt=".2f")
        st.pyplot(plt.gcf(), use_container_width=True)
        plt.clf()
    with col10:
        st.markdown('<div class="centered-subheader">SHAP Waterfall Plot for Selected Prediction</div>', unsafe_allow_html=True)
        X = df.drop(columns=['CKD_Status']) 
        row_index = st.slider("Select Row Index", 0, len(X)-1, 0)
        sample = X.iloc[[row_index]]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value, shap_values[0], sample.iloc[0], show=False)
        st.pyplot(fig)

st.divider()

st.markdown("""
    <div class="info-block">
        <div class="info-title">Model Accuracy</div>
        <p>Demonstrate how well the Model does with respect to the Dataset and Explaination of terms such as Accuracy, Precision, F1 Score.</p>
        <div class="info-subtitle">Note: The observed accuracy is due to it's strong feature seperability in the dataset.</div>
    </div>
""", unsafe_allow_html=True)

with st.expander("__Model Evaluation Metrics__"):
    col11, col12= st.columns([1,2])
    maps={
    0: 'Non-CKD',
    1: 'CKD'
    }
    with col11:
        st.markdown('<div class="centered-subheader_matrix">Confusion Matrix</div>', unsafe_allow_html=True)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=maps.values())
        disp.plot(cmap='Blues')
        st.pyplot(disp.figure_, use_container_width=True)
    with col12:
        st.markdown('<div class="centered-subheader_matrix">Classification Report</div>', unsafe_allow_html=True)
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)

st.divider()

st.markdown("""
<div class="info-block">
    <div class="info-title">Key Takeaways</div>
    <ul>
        <li><b>Kidney function markers</b> such as GFR, BUN, and Creatinine show strong and consistent influence on CKD prediction.</li>
        <li><b>Feature distributions</b> reveal clear separation patterns between healthy and high-risk cases.</li>
        <li><b>Model explanations (SHAP)</b> confirm that predictions are driven by clinically meaningful signals rather than noise.</li>
        <li><b>Performance metrics</b> indicate high accuracy on the evaluation set, though results depend on dataset scope.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.divider()







