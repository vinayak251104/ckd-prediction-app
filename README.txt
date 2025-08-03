CKD PREDICTION APP

This is a multipage Streamlit web app that predicts Chronic Kidney Disease (CKD) risk based on clinical input. It uses an XGBoost model trained on a synthetic dataset. The app also includes SHAP explainability and performance metrics.

-----------------------------
LIVE DEMO
-----------------------------
Streamlit App: https://vinayak251104-ckd-prediction-app.streamlit.app/

-----------------------------
APP STRUCTURE
-----------------------------
1. DASHBOARD - Feature distributions  
2. PREDICTION - User form and CKD risk output  
3. ANALYSIS - SHAP plot, confusion matrix, and evaluation metrics  

-----------------------------
DATASET
-----------------------------
**Features**: Age, Creatinine Level, BUN, Diabetes, Hypertension, GFR, Urine Output, Dialysis Needed  
**Target**: CKD Status (0 = No CKD, 1 = CKD)  

-----------------------------
MODEL INFO
-----------------------------
**Model**: XGBoost Classifier  
**Training Split**: 80% Train / 20% Test  
**Metrics on test data**:  
- Accuracy: 100%  
- F1 Score (CKD): 1.0  
- SHAP plots used for feature contribution explanation  

-----------------------------
FUZZY DATA ROBUSTNESS TESTING
-----------------------------
Model was tested on randomized synthetic samples with controlled noise to test generalization:

- **Noise scale 0.01 to 0.05** → Accuracy: 85% to 89%  
- **Noise scale > 1** → Accuracy drops sharply (down to 50% at extreme values)  

This shows the model is reliable under mild perturbations but fragile under heavy noise.

-----------------------------
HOW TO RUN LOCALLY
-----------------------------
1. Clone the repository  
2. Navigate to the project folder  
3. Install required packages  
4. Run with Streamlit  

**Commands**:  
git clone https://github.com/vinayak251104/ckd-prediction-app.git  
cd ckd-prediction-app  
pip install -r requirements.txt  
streamlit run Main.py  

-----------------------------
FILE STRUCTURE
-----------------------------
Main.py  
pages/Prediction.py  
pages/Analysis.py  
final_xg_model_for_ckd_status.pkl  
kidney_disease_dataset.csv  
requirements.txt  

-----------------------------
AUTHOR
-----------------------------
Developed by: Vinayak (vinayak251104)

