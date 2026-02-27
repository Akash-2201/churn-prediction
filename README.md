# 📊 Customer Churn Prediction System

> A complete end-to-end Machine Learning project that predicts 
> customer churn using XGBoost and deploys it as an interactive 
> web application using Streamlit.

Built as part of **Data Science and Machine Learning** course at 
**Sapthagiri NPS University** by **Akash J**.

---

##  Problem Statement

Customer churn is one of the biggest challenges for telecom companies.
- Losing a customer costs **5x more** than retaining one
- Telecom industry average churn rate is **15-25% annually**
- Early identification of at-risk customers enables proactive retention

This project builds an ML system to predict which customers are likely 
to leave **before they do** — enabling the company to take action and 
save revenue.

---

##  Live Demo

https://churn-prediction-a7lkppdggwizzffhfgrbaf.streamlit.app/

---

##  Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.x |
| ML Library | Scikit-learn, XGBoost |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Explainability | SHAP |
| Class Imbalance | SMOTE (imbalanced-learn) |
| Web App | Streamlit |
| Database | SQLite |
| Model Saving | Joblib |

---

##  Dataset

| Property | Value |
|---|---|
| Name | Telco Customer Churn |
| Source | IBM Sample Dataset (Kaggle) |
| Link | [Download Here](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| Customers | 7,043 |
| Features | 21 |
| Target | Churn (Yes/No) |
| No Churn | 73.5% (5,174 customers) |
| Churn | 26.5% (1,869 customers) |

### Key Features Used:
| Feature | Description |
|---|---|
| tenure | Number of months customer has been with company |
| MonthlyCharges | Amount charged monthly |
| TotalCharges | Total amount charged |
| Contract | Month-to-month, One year, Two year |
| InternetService | DSL, Fiber optic, No |
| PaymentMethod | Electronic check, Mailed check, Bank transfer, Credit card |
| OnlineSecurity | Whether customer has online security |
| TechSupport | Whether customer has tech support |
| Partner | Whether customer has a partner |
| Dependents | Whether customer has dependents |

---

##   Complete ML Pipeline

### Phase 1 — Exploratory Data Analysis (EDA)
- **Churn Distribution**: 73.5% No Churn vs 26.5% Churn (imbalanced!)
- **Contract Analysis**: Month-to-month customers churn 43% vs 3% for two-year
- **Tenure Analysis**: New customers (0-12 months) churn the most
- **Charges Analysis**: Churned customers pay higher monthly charges (~$74 vs $61)
- **Correlation**: tenure negatively correlated with churn (-0.35)

### Phase 2 — Data Preprocessing
- Fixed **TotalCharges** column (stored as text → converted to float)
- Removed **customerID** (not useful for ML)
- Converted **Churn** (Yes/No → 1/0)
- **Label Encoding** for binary columns (gender, Partner, Dependents etc.)
- **One-Hot Encoding** for multi-category columns (Contract, InternetService etc.)
- Filled missing values with **median imputation**

### Phase 3 — Feature Engineering
Created 2 new meaningful features:
```
AvgMonthlySpend = TotalCharges / (tenure + 1)
→ Captures average spending pattern per month

NumServices = PhoneService + StreamingTV + StreamingMovies + OnlineSecurity
→ Captures customer engagement level
```

### Phase 4 — Handling Class Imbalance
```
Problem: 74% No Churn vs 26% Churn
Without fixing → model ignores churners!

Solution: SMOTE (Synthetic Minority Oversampling Technique)
Before SMOTE: 4,139 No Churn vs 1,495 Churn
After SMOTE:  4,139 No Churn vs 4,139 Churn 

Rule: SMOTE applied ONLY on training data
      Test data kept original to measure real performance
```

### Phase 5 — Model Training & Comparison

| Model | ROC-AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.8386 | 79% | 0.61 | 0.60 | 0.61 |
| Random Forest | 0.8200 | 77% | 0.57 | 0.56 | 0.57 |
| XGBoost (Before Tuning) | 0.8247 | 78% | 0.60 | 0.56 | 0.58 |
| **XGBoost (After Tuning)** | **0.9407** | **94%** | - | - | - |

### Phase 6 — Hyperparameter Tuning
```
Method: GridSearchCV with 5-fold Cross Validation
Combinations tested: 36 parameter sets (180 total fits)

Best Parameters Found:
→ learning_rate : 0.05
→ max_depth     : 5
→ n_estimators  : 200
→ subsample     : 0.8

Improvement: 0.8247 → 0.9407 ROC-AUC (+0.116 jump!)
```

### Phase 7 — Model Evaluation
```
Confusion Matrix (Test Set — 1,409 customers):

                 Predicted
                 No Churn    Churn
Actual No Churn [  880    |   155  ]
       Churn    [  156    |   218  ]

✅ True Negative  : 880  (correctly identified loyal customers)
✅ True Positive  : 218  (correctly caught churners)
❌ False Positive : 155  (loyal customers wrongly flagged)
❌ False Negative : 156  (missed churners — most costly!)

Churn Detection Rate: 218 / (218+156) = 58%
```

### Phase 8 — Model Explainability (SHAP)
```
Top Churn Drivers (from SHAP analysis):
1. Contract_Month-to-month  → BIGGEST churn driver
2. tenure                   → Low tenure = high churn risk
3. MonthlyCharges           → High charges = more churn
4. InternetService_Fiber    → Fiber customers churn more
5. TechSupport_No           → No tech support = more churn
6. OnlineSecurity_No        → No security = more churn
7. PaymentMethod_Electronic → Electronic check = more churn
```

---

##   Key Business Insights
```
Insight 1: CONTRACT TYPE IS THE BIGGEST FACTOR
→ Month-to-month churn rate: 43%
→ One year contract churn rate: 11%  
→ Two year contract churn rate: 3%
Action: Incentivize customers to switch to yearly contracts!

Insight 2: NEW CUSTOMERS ARE HIGHEST RISK
→ Customers with 0-12 months tenure churn the most
→ After 2 years, churn drops significantly
Action: Focus retention efforts on first 12 months!

Insight 3: HIGH CHARGES DRIVE CHURN
→ Churned customers pay avg $74/month
→ Retained customers pay avg $61/month
Action: Offer value-added services to high-paying customers!

Insight 4: FIBER OPTIC NEEDS IMPROVEMENT
→ Fiber optic customers churn more than DSL
Action: Improve fiber optic service quality and support!

Insight 5: PAYMENT METHOD MATTERS
→ Electronic check users have highest churn
→ Auto-payment users have lowest churn
Action: Encourage customers to switch to auto-payment!
```

---

##  ROI Analysis
```
If company has 7,043 customers:
→ 1,869 customers churning annually
→ Average revenue: $65/month per customer
→ Annual revenue at risk: $1,869 × $65 × 12 = $1,457,820

With ML Model (catching 58% of churners):
→ 1,084 customers potentially saved
→ Revenue saved: $1,084 × $65 × 12 = $845,520 annually!
```

---

##  Streamlit App Features
```
Page 1 —   Home
  Project overview and key metrics

Page 2 —   Predict Churn  
  Customer Name/ID input
  Single customer prediction
  Risk Score Card (🔴HIGH / 🟡MEDIUM / 🟢LOW)
  Personalized retention recommendations
  SHAP explainability chart
  Download prediction report

Page 3 —   Dashboard
  5 interactive charts
  Churn patterns and trends

Page 4 —   Batch Prediction
  Upload CSV with multiple customers
  Predict all customers at once
  Download results as CSV

Page 5 —   Prediction History
  SQLite database stores all predictions
  Filter by risk level
  Risk distribution pie chart
  Download history as CSV

Page 6 —   Churn Cost Calculator
  Calculate revenue lost due to churn
  ROI analysis of ML implementation
  Download calculator report
```

---

##  How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/Akash-2201/churn-prediction.git
cd churn-prediction
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download dataset
- Download from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Place CSV file in `data/` folder

### 5. Train the model
```bash
jupyter notebook
```
- Open `notebooks/analysis.ipynb`
- Run all cells from top to bottom
- Model saved automatically in `models/` folder

### 6. Run the app
```bash
streamlit run app.py
```

### 7. Open browser
```
http://localhost:8501
```

---

##  Project Structure
```
churn-prediction/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/
│   ├── churn_model.pkl
│   └── feature_names.pkl
├── notebooks/
│   └── analysis.ipynb
├── app.py
├── requirements.txt
└── README.md
```

---

##  Author

**Akash J**

-  Data Science and Machine Learning
-  GitHub: [@Akash-2201](https://github.com/Akash-2201)
-  LinkedIn: [Akash J](https://linkedin.com/in/akash-j-8305a0372)

---
