A multipage Streamlit web app that predicts Chronic Kidney Disease (CKD) using clinical features like GFR, BUN, Creatinine, and more. Powered by a tuned XGBoost model, the app provides:

- Risk prediction from user input  
- Dataset analysis via feature distributions  
- SHAP-based model explainability  
- Confusion matrix and classification metrics  

### Robustness Testing on Fuzzy Data

The model achieved ~100% accuracy on validation data, suggesting potential overfitting. To verify robustness, I tested on synthetic data with noise:

- Noise scale **0.01–0.05** → Accuracy: **~85–89%**  
- Noise scale **0.05–0.2** → Accuracy: **~75–85%**

### Live Demo

- Link for the App (deployed on Streamlit Community Cloud) [https://vinayak251104-ckd-prediction-app.streamlit.app/]


### Run Locally

```bash
git clone https://github.com/vinayak251104/ckd-prediction-app.git  
cd ckd-prediction-app  
pip install -r requirements.txt  
streamlit run Main.py








