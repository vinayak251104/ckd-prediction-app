-----------------------------------------------------
CKD PREDICTOR MULTIPAGE APP (STREAMLIT)
-----------------------------------------------------

DESCRIPTION
-----------
This project is a full-fledged Chronic Kidney Disease (CKD) prediction dashboard built using Python and Streamlit. It combines a trained machine learning model (XGBoost) with an interactive user interface that enables users to:

- Enter clinical test values and receive real-time CKD risk predictions.
- Analyze dataset distributions and SHAP explanations.
- Understand the model’s performance through metrics and visualization.

It is structured as a multipage Streamlit app with three pages:
1. Dashboard (Overview + Data Info)
2. Prediction Page (CKD Risk Estimator)
3. Analysis Page (SHAP, Confusion Matrix, Metrics)

-----------------------------------------------------
DATASET
-------
Source: Simulated kidney disease dataset.

Features used:
- Age
- Creatinine_Level
- BUN (Blood Urea Nitrogen)
- Diabetes (Yes/No)
- Hypertension (Yes/No)
- GFR (Glomerular Filtration Rate)
- Urine Output
- Dialysis Needed (Yes/No)

Target:
- CKD_Status (0 = No CKD, 1 = CKD)

-----------------------------------------------------
MODEL INFO
----------
- Model Used: XGBoost Classifier
- Training Strategy: 80/20 Train-Test Split
- Hyperparameters: Tuned for max depth, learning rate, estimators
- Feature Importance: Visualized with SHAP

-----------------------------------------------------
PREDICTION LOGIC
----------------
User inputs values for the 8 features through a Streamlit form. Upon submission:
- Categorical features are mapped to binary (Yes = 1, No = 0)
- A prediction probability is generated using the saved model
- An animated gauge displays CKD risk level
- Result text is displayed along with user input summary

-----------------------------------------------------
MODEL PERFORMANCE
-----------------
- Accuracy on real test data (20% holdout): 100%
  (May indicate mild overfitting due to high model confidence)

- Accuracy on randomized synthetic data (100 samples): 79.1%
  (Better indicator of generalization — realistic variation introduced)

- F1 Score (CKD): 0.80
- Balanced precision and recall (~0.79 each)

-----------------------------------------------------
VISUALIZATION FEATURES
-----------------------
✅ Histograms & KDE plots for feature distribution  
✅ Heatmap of feature correlation  
✅ SHAP waterfall plot (row-specific feature impact)  
✅ Confusion matrix with label mapping  
✅ Classification report (precision, recall, f1-score)

-----------------------------------------------------
FILE STRUCTURE
--------------
📂 kidney_ckd_app/
├── dashboard.py             # Overview page
├── predictor.py             # User input form + prediction logic
├── analysis.py              # SHAP, metrics, confusion matrix
├── final_xg_model_for_ckd_status.pkl
├── kidney_disease_dataset.csv
├── README.txt

CREDITS
-------
Developed by: Vinayak (vinayak251104)  
Modeling + App Design + UI: Solo  
ML Help: ChatGPT (for review, refactor, and debugging)

-----------------------------------------------------
