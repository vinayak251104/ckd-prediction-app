# CKD Prediction App (NephroCheck)

A multi-page Streamlit web application for predicting Chronic Kidney Disease (CKD) using clinically relevant
features such as GFR, BUN, Creatinine, Age, and comorbid conditions.

The system is powered by a tuned XGBoost classifier and focuses on both prediction and interpretability.

🔗 **Live App:** https://vinayak251104-ckd-prediction-app.streamlit.app/

> ⚠️ This project is for demonstration purposes only. It is not a medical diagnostic tool.

## Key Features

* Risk prediction based on user-provided clinical parameters
* Interactive data analysis using feature distribution visualizations
* Model explainability via SHAP-based feature attribution
* Model evaluation using confusion matrix and classification metrics
* Adjustable decision threshold to control sensitivity vs. specificity

## Dataset Overview

The dataset contains clinical measurements related to kidney function and associated health conditions.

* **Source:** Kaggle – Kidney Disease Risk Dataset
* **Total records:** 2304
* **Total features:** 9
* **Note:** The CKD labels appear rule-generated rather than clinician-assigned (see below), so the data is
  treated as synthetic for evaluation purposes.

Features:

* Glomerular Filtration Rate (GFR)
* Blood Urea Nitrogen (BUN)
* Creatinine Level
* Urine Output
* Age
* Diabetes, Hypertension, Dialysis status

## Model Performance & Robustness

On the held-out validation split, the model achieves near-perfect accuracy. Analysis of the dataset shows
why: the CKD labels follow a clean threshold pattern (roughly GFR < 60, BUN > 30, or creatinine > 2.5), and
a depth-3 decision tree reproduces them with 100% cross-validated accuracy. The perfect scores therefore show
that the model learned this pattern. They should not be read as real-world clinical accuracy.

Age, diabetes, hypertension and urine output carry almost no predictive signal in this dataset
(~50% single-feature accuracy, i.e. chance). The model's most influential features are GFR, BUN and creatinine.

To probe robustness, the model was also evaluated on synthetic noisy inputs:

| Noise scale | Accuracy |
|---|---|
| 0.01 – 0.05 | ~85–89% |
| 0.05 – 0.2 | ~75–85% |


## Run Locally

```bash
pip install -r requirements.txt
streamlit run Main.py
```

## Tech Stack

Python, Streamlit, XGBoost, SHAP, scikit-learn, pandas, matplotlib, seaborn




