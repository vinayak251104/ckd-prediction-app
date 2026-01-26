# CKD Prediction App

A multi-page **Streamlit web application** for predicting **Chronic Kidney Disease (CKD)** using clinically relevant features such as **GFR, BUN, Creatinine, Age, and comorbid conditions**.  
The system is powered by a **tuned XGBoost classifier** and focuses on both prediction and interpretability.

## Key Features

- **Risk prediction** based on user-provided clinical parameters  
- **Interactive data analysis** using feature distribution visualizations  
- **Model explainability** via SHAP-based feature attribution  
- **Model evaluation** using confusion matrix and classification metrics  
- **Adjustable decision threshold** to control sensitivity vs. specificity  

## Model Performance & Robustness

On the held-out validation split, the model achieves **near-perfect accuracy**, largely due to the strong discriminative power of key clinical indicators in the dataset.

To assess robustness beyond clean validation data, the model was further evaluated using **synthetic noisy inputs**:

- **Noise scale 0.01–0.05** → Accuracy: **~85–89%**  
- **Noise scale 0.05–0.2** → Accuracy: **~75–85%**

These results indicate that while the model performs exceptionally well on structured clinical data, performance degrades gracefully under increasing noise—suggesting reasonable robustness but highlighting the importance of validation on diverse, real-world datasets.

## Live Demo

🔗 **Live App:**  
https://vinayak251104-ckd-prediction-app.streamlit.app/





