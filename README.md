# 📊 Customer Churn Prediction System

A Machine Learning web app that predicts customer churn using XGBoost.
Built as part of **Data Science and Machine Learning** .

---
##   Problem Statement

Customer churn is one of the biggest challenges for telecom companies. 
Losing a customer costs 5x more than retaining one. 

This project builds an ML model to predict which customers are likely 
to leave — before they do — so the company can take proactive action 
and save revenue.


##  Live Demo
https://churn-prediction-a7lkppdggwizzffhfgrbaf.streamlit.app/

---

##  Tech Stack
Python, XGBoost, Streamlit, SHAP, SQLite, Pandas, Plotly

---

##  Dataset
Telco Customer Churn — 7,043 customers, 21 features (Kaggle)

---

##   App Features
-   Single Customer Churn Prediction
-   Batch Prediction (CSV Upload)
-   Interactive Dashboard
-   Prediction History (SQLite Database)
-   Churn Cost Calculator
-   SHAP Explainability
-   Download Reports

---

##  Model Performance
| Model | ROC-AUC |
|---|---|
| Logistic Regression | 0.84 |
| Random Forest | 0.82 |
| **XGBoost (Final)** | **0.94 ** |

---

##  How to Run
```bash
git clone https://github.com/Akash-2201/churn-prediction.git
cd churn-prediction
pip install -r requirements.txt
streamlit run app.py
```

---

## 👨‍💻 Author
**Akash J** — Sapthagiri NPS University
- 🐙 [GitHub](https://github.com/Akash-2201)
- 💼 [LinkedIn](https://linkedin.com/in/akash-j-8305a0372)
