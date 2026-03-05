import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import sqlite3
from datetime import datetime

# ── Database Setup ──
def init_db():
    conn = sqlite3.connect('churn_history.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            time TEXT,
            customer_name TEXT,
            tenure INTEGER,
            monthly_charges REAL,
            contract TEXT,
            internet TEXT,
            payment TEXT,
            senior TEXT,
            partner TEXT,
            churn_probability REAL,
            risk_level TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_prediction(customer_name, tenure, monthly_charges, contract,
                    internet, payment, senior, partner,
                    prob, risk_level):
    conn = sqlite3.connect('churn_history.db')
    c = conn.cursor()
    now = datetime.now()
    c.execute('''
        INSERT INTO predictions
        (date, time, customer_name, tenure, monthly_charges, contract,
         internet, payment, senior, partner,
         churn_probability, risk_level)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        now.strftime("%Y-%m-%d"),
        now.strftime("%H:%M:%S"),
        str(customer_name),
        int(tenure), float(monthly_charges), str(contract),
        str(internet), str(payment), str(senior), str(partner),
        float(round(prob * 100, 2)), str(risk_level)
    ))
    conn.commit()
    conn.close()

def load_history():
    conn = sqlite3.connect('churn_history.db')
    df = pd.read_sql_query(
        "SELECT * FROM predictions ORDER BY id DESC", conn)
    conn.close()
    df['risk_level'] = df['risk_level'].astype(str)
    df['contract'] = df['contract'].astype(str)
    df['internet'] = df['internet'].astype(str)
    df['payment'] = df['payment'].astype(str)
    df['senior'] = df['senior'].astype(str)
    df['partner'] = df['partner'].astype(str)
    df['date'] = df['date'].astype(str)
    df['time'] = df['time'].astype(str)
    df['churn_probability'] = df['churn_probability'].astype(float)
    df['tenure'] = df['tenure'].astype(int)
    df['monthly_charges'] = df['monthly_charges'].astype(float)
    return df

def delete_all():
    conn = sqlite3.connect('churn_history.db')
    c = conn.cursor()
    c.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()

# Initialize database
init_db()

st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# ── Load Model ──
@st.cache_resource
def load_model():
    from xgboost import XGBClassifier
    model = XGBClassifier()
    model.load_model('models/churn_model.json')
    features = joblib.load('models/feature_names.pkl')
    return model, features

model, feature_names = load_model()

# ── Sidebar Navigation ──
st.sidebar.title("📊 Churn Predictor")
st.sidebar.write("---")
page = st.sidebar.selectbox(
    "Go to Page",
    ["🏠 Home", "🔍 Predict Churn", "📊 Dashboard",
     "📂 Batch Prediction", "🗄️ Prediction History",
     "💰 Churn Cost Calculator"]
)

# ════════════════════════════════
# 🏠 HOME PAGE
# ════════════════════════════════
if page == "🏠 Home":
    st.title("📊 Customer Churn Prediction App")
    st.write("#### Predict which customers are likely to leave — before they do!")
    st.write("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", "7,043")
    col2.metric("Churn Rate", "26.5%")
    col3.metric("Model Used", "XGBoost")
    col4.metric("ROC-AUC Score", "0.94")

    st.write("---")
    st.write("### How to Use This App")
    st.info("1. Go to Predict Churn → Enter customer details → Get prediction")
    st.info("2. Go to Dashboard → See overall churn trends and patterns")

    st.write("---")
    st.write("### About the Project")
    st.write("""
    This project builds a complete ML pipeline:
    - **Data**: Telco Customer Churn dataset (7,043 customers, 21 features)
    - **Model**: XGBoost with hyperparameter tuning
    - **Techniques**: SMOTE, SHAP Explainability, Feature Engineering
    - **Deployment**: Streamlit Cloud
    """)

# ════════════════════════════════
# 🔍 PREDICT PAGE
# ════════════════════════════════
elif page == "🔍 Predict Churn":
    st.title("🔍 Customer Churn Predictor")
    st.write("Fill in the customer details below and click Predict.")
    st.write("---")
    customer_name = st.text_input("👤 Customer Name or ID",
                                   placeholder="Enter customer name or ID...")
    st.write("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("#### 📋 Account Info")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.number_input("Monthly Charges ($)", 20.0, 120.0, 65.0)
        total_charges = monthly_charges * tenure
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])

    with col2:
        st.write("#### 📄 Contract Info")
        contract = st.selectbox("Contract Type",
            ["Month-to-month", "One year", "Two year"])
        payment = st.selectbox("Payment Method",
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)",
             "Credit card (automatic)"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

    with col3:
        st.write("#### 🌐 Services")
        internet = st.selectbox("Internet Service",
            ["Fiber optic", "DSL", "No"])
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        phone = st.selectbox("Phone Service", ["Yes", "No"])

    st.write("---")

    if st.button("🎯 Predict Churn", type="primary", use_container_width=True):

        input_dict = {col: 0 for col in feature_names}

        input_dict['tenure'] = tenure
        input_dict['MonthlyCharges'] = monthly_charges
        input_dict['TotalCharges'] = float(total_charges)
        input_dict['SeniorCitizen'] = 1 if senior == "Yes" else 0
        input_dict['Partner'] = 1 if partner == "Yes" else 0
        input_dict['Dependents'] = 1 if dependents == "Yes" else 0
        input_dict['PhoneService'] = 1 if phone == "Yes" else 0
        input_dict['PaperlessBilling'] = 1 if paperless == "Yes" else 0
        input_dict['AvgMonthlySpend'] = float(total_charges) / (tenure + 1)

        contract_map = {
            "Month-to-month": "Contract_Month-to-month",
            "One year": "Contract_One year",
            "Two year": "Contract_Two year"
        }
        if contract_map[contract] in input_dict:
            input_dict[contract_map[contract]] = 1

        internet_map = {
            "Fiber optic": "InternetService_Fiber optic",
            "DSL": "InternetService_DSL",
            "No": "InternetService_No"
        }
        if internet_map[internet] in input_dict:
            input_dict[internet_map[internet]] = 1

        payment_map = {
            "Electronic check": "PaymentMethod_Electronic check",
            "Mailed check": "PaymentMethod_Mailed check",
            "Bank transfer (automatic)": "PaymentMethod_Bank transfer (automatic)",
            "Credit card (automatic)": "PaymentMethod_Credit card (automatic)"
        }
        if payment_map[payment] in input_dict:
            input_dict[payment_map[payment]] = 1

        input_dict['NumServices'] = (
            input_dict.get('PhoneService', 0) +
            input_dict.get('StreamingTV_Yes', 0) +
            input_dict.get('StreamingMovies_Yes', 0) +
            input_dict.get('OnlineSecurity_Yes', 0)
        )

        input_df = pd.DataFrame([input_dict])
        prob = model.predict_proba(input_df)[0][1]

        if prob >= 0.70:
            risk_level = "HIGH"
        elif prob >= 0.40:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        save_prediction(
            customer_name, tenure, monthly_charges, contract,
            internet, payment, senior, partner,
            prob, risk_level
        )

        st.write(f"## 🎯 Prediction Result for: {customer_name if customer_name else 'Unknown Customer'}")
        col_a, col_b = st.columns(2)

        with col_a:
            if prob >= 0.70:
                st.error("🔴 HIGH RISK CUSTOMER")
            elif prob >= 0.40:
                st.warning("🟡 MEDIUM RISK CUSTOMER")
            else:
                st.success("🟢 LOW RISK CUSTOMER")

            st.metric("Churn Probability", f"{prob*100:.1f}%")
            st.metric("Risk Level", risk_level)

            st.write("### 💡 Recommendations:")
            if risk_level == "HIGH":
                st.error("1. Call customer immediately")
                st.error("2. Offer 20% loyalty discount")
                st.error("3. Upgrade to yearly contract")
                st.error("4. Assign dedicated support agent")
            elif risk_level == "MEDIUM":
                st.warning("1. Send retention email")
                st.warning("2. Offer free service upgrade")
                st.warning("3. Check satisfaction score")
            else:
                st.success("1. Customer is happy!")
                st.success("2. Send thank you message")
                st.success("3. Offer referral bonus")

        with col_b:
            fig = px.bar(
                x=["Stay", "Churn"],
                y=[float(1 - prob), float(prob)],
                color=["Stay", "Churn"],
                color_discrete_map={"Stay": "#2ecc71", "Churn": "#e74c3c"},
                title="Prediction Probability"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.write("---")
        st.write("### 📥 Download Prediction Report")

        report = f"""
CUSTOMER CHURN PREDICTION REPORT
==================================
Tenure          : {tenure} months
Monthly Charges : ${monthly_charges}
Total Charges   : ${total_charges}
Contract Type   : {contract}
Internet Service: {internet}
Payment Method  : {payment}
Senior Citizen  : {senior}
Partner         : {partner}
Dependents      : {dependents}

PREDICTION RESULT
==================================
Churn Probability : {prob*100:.1f}%
Risk Level        : {risk_level}

RECOMMENDATIONS
==================================
"""
        if risk_level == "HIGH":
            report += """1. Call customer immediately
2. Offer 20% loyalty discount
3. Upgrade to yearly contract
4. Assign dedicated support agent"""
        elif risk_level == "MEDIUM":
            report += """1. Send retention email
2. Offer free service upgrade
3. Check satisfaction score"""
        else:
            report += """1. Customer is happy!
2. Send thank you message
3. Offer referral bonus"""

        st.download_button(
            label="📥 Download Report as TXT",
            data=report,
            file_name="churn_prediction_report.txt",
            mime="text/plain"
        )

        st.write("---")
        st.write("### 🧠 Why did the model predict this? (SHAP)")
        try:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(input_df)
            fig2, ax = plt.subplots(figsize=(10, 4))
            shap.summary_plot(shap_vals, input_df,
                              plot_type='bar', max_display=10, show=False)
            st.pyplot(fig2)
            plt.close()
        except Exception as e:
            st.warning("SHAP visualization not available in cloud deployment.")

# ════════════════════════════════
# 📊 DASHBOARD PAGE
# ════════════════════════════════
elif page == "📊 Dashboard":
    st.title("📊 Churn Analysis Dashboard")
    st.write("Insights from the Telco Customer Churn dataset")
    st.write("---")

    try:
        @st.cache_data
        def load_data():
            return pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

        df = load_data()

        col1, col2 = st.columns(2)

        with col1:
            st.write("### Churn by Contract Type")
            fig1 = px.histogram(df, x='Contract', color='Churn',
                                barmode='group',
                                color_discrete_map={'No':'#2ecc71','Yes':'#e74c3c'})
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.write("### Monthly Charges vs Churn")
            fig2 = px.box(df, x='Churn', y='MonthlyCharges',
                          color='Churn',
                          color_discrete_map={'No':'#2ecc71','Yes':'#e74c3c'})
            st.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            st.write("### Tenure Distribution by Churn")
            fig3 = px.histogram(df, x='tenure', color='Churn', nbins=30,
                                color_discrete_map={'No':'#2ecc71','Yes':'#e74c3c'})
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            st.write("### Churn by Internet Service")
            fig4 = px.histogram(df, x='InternetService', color='Churn',
                                barmode='group',
                                color_discrete_map={'No':'#2ecc71','Yes':'#e74c3c'})
            st.plotly_chart(fig4, use_container_width=True)

        st.write("### Churn by Payment Method")
        fig5 = px.histogram(df, x='PaymentMethod', color='Churn',
                            barmode='group',
                            color_discrete_map={'No':'#2ecc71','Yes':'#e74c3c'})
        st.plotly_chart(fig5, use_container_width=True)

    except FileNotFoundError:
        st.error("Dataset not found! Place the CSV file in the data/ folder.")

# ════════════════════════════════
# 📂 BATCH PREDICTION PAGE
# ════════════════════════════════
elif page == "📂 Batch Prediction":
    st.title("📂 Batch Customer Churn Prediction")
    st.write("Upload a CSV file with multiple customers — get full analysis, charts and revenue impact!")
    st.write("---")

    st.write("### 📋 Required CSV Format:")
    sample_data = pd.DataFrame({
        'tenure': [12, 24, 5],
        'MonthlyCharges': [65.0, 45.0, 90.0],
        'TotalCharges': [780.0, 1080.0, 450.0],
        'Contract': ['Month-to-month', 'One year', 'Two year'],
        'InternetService': ['Fiber optic', 'DSL', 'No'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)'],
        'SeniorCitizen': [0, 1, 0],
        'Partner': ['Yes', 'No', 'Yes'],
        'Dependents': ['No', 'Yes', 'No'],
        'PhoneService': ['Yes', 'Yes', 'No'],
        'PaperlessBilling': ['Yes', 'No', 'Yes']
    })
    st.dataframe(sample_data)

    st.download_button(
        label="📥 Download Sample CSV Template",
        data=sample_data.to_csv(index=False),
        file_name="sample_customers.csv",
        mime="text/csv"
    )

    st.write("---")
    st.write("### 📤 Upload Your CSV File:")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        upload_df = pd.read_csv(uploaded_file)
        st.write(f"✅ File uploaded successfully! **{len(upload_df)} customers** found.")
        st.dataframe(upload_df)

        if st.button("🎯 Predict Churn for All Customers", type="primary", use_container_width=True):
            results = []

            progress = st.progress(0)
            status = st.empty()

            for idx, row in upload_df.iterrows():
                status.text(f"Processing customer {idx+1} of {len(upload_df)}...")
                progress.progress((idx + 1) / len(upload_df))

                input_dict = {col: 0 for col in feature_names}

                input_dict['tenure'] = row.get('tenure', 0)
                input_dict['MonthlyCharges'] = row.get('MonthlyCharges', 0)
                input_dict['TotalCharges'] = row.get('TotalCharges', 0)
                input_dict['SeniorCitizen'] = row.get('SeniorCitizen', 0)
                input_dict['Partner'] = 1 if row.get('Partner') == 'Yes' else 0
                input_dict['Dependents'] = 1 if row.get('Dependents') == 'Yes' else 0
                input_dict['PhoneService'] = 1 if row.get('PhoneService') == 'Yes' else 0
                input_dict['PaperlessBilling'] = 1 if row.get('PaperlessBilling') == 'Yes' else 0
                input_dict['AvgMonthlySpend'] = row.get('TotalCharges', 0) / (row.get('tenure', 1) + 1)

                contract_val = row.get('Contract', '')
                contract_map = {
                    "Month-to-month": "Contract_Month-to-month",
                    "One year": "Contract_One year",
                    "Two year": "Contract_Two year"
                }
                if contract_val in contract_map and contract_map[contract_val] in input_dict:
                    input_dict[contract_map[contract_val]] = 1

                internet_val = row.get('InternetService', '')
                internet_map = {
                    "Fiber optic": "InternetService_Fiber optic",
                    "DSL": "InternetService_DSL",
                    "No": "InternetService_No"
                }
                if internet_val in internet_map and internet_map[internet_val] in input_dict:
                    input_dict[internet_map[internet_val]] = 1

                payment_val = row.get('PaymentMethod', '')
                payment_map = {
                    "Electronic check": "PaymentMethod_Electronic check",
                    "Mailed check": "PaymentMethod_Mailed check",
                    "Bank transfer (automatic)": "PaymentMethod_Bank transfer (automatic)",
                    "Credit card (automatic)": "PaymentMethod_Credit card (automatic)"
                }
                if payment_val in payment_map and payment_map[payment_val] in input_dict:
                    input_dict[payment_map[payment_val]] = 1

                input_df = pd.DataFrame([input_dict])
                prob = model.predict_proba(input_df)[0][1]
                batch_risk = "HIGH" if prob >= 0.50 else "MEDIUM" if prob >= 0.30 else "LOW"

                save_prediction(
                    f"Batch Customer {idx+1}",
                    row.get('tenure', 0),
                    row.get('MonthlyCharges', 0),
                    row.get('Contract', ''),
                    row.get('InternetService', ''),
                    row.get('PaymentMethod', ''),
                    str(row.get('SeniorCitizen', 0)),
                    row.get('Partner', ''),
                    prob,
                    batch_risk
                )

                # Recommendations
                if batch_risk == "HIGH":
                    recommendation = "Call immediately + Offer 20% discount"
                elif batch_risk == "MEDIUM":
                    recommendation = "Send retention email + Free upgrade"
                else:
                    recommendation = "Send thank you + Referral bonus"

                results.append({
                    'Customer #': idx + 1,
                    'Tenure (months)': row.get('tenure', 0),
                    'Monthly Charges ($)': row.get('MonthlyCharges', 0),
                    'Contract': row.get('Contract', ''),
                    'Internet': row.get('InternetService', ''),
                    'Payment': row.get('PaymentMethod', ''),
                    'Churn Probability (%)': round(prob * 100, 1),
                    'Risk Level': batch_risk,
                    'Recommendation': recommendation
                })

            progress.empty()
            status.empty()

            results_df = pd.DataFrame(results)

            # ── SECTION 1: Summary Metrics ──
            st.write("---")
            st.write("## 📊 Prediction Summary")
            col1, col2, col3, col4 = st.columns(4)
            total = len(results_df)
            high_risk = len(results_df[results_df['Risk Level'] == 'HIGH'])
            medium_risk = len(results_df[results_df['Risk Level'] == 'MEDIUM'])
            low_risk = len(results_df[results_df['Risk Level'] == 'LOW'])

            col1.metric("Total Customers", total)
            col2.metric("🔴 High Risk", high_risk)
            col3.metric("🟡 Medium Risk", medium_risk)
            col4.metric("🟢 Low Risk", low_risk)

            # ── SECTION 2: Prediction Table ──
            st.write("---")
            st.write("### 📋 Detailed Predictions")

            risk_filter = st.selectbox(
                "Filter by Risk Level",
                ["All", "HIGH", "MEDIUM", "LOW"]
            )
            if risk_filter != "All":
                display_df = results_df[results_df['Risk Level'] == risk_filter]
            else:
                display_df = results_df

            st.dataframe(display_df, use_container_width=True)

            # ── SECTION 3: Charts ──
            st.write("---")
            st.write("### 📈 Visual Analysis")

            col_c1, col_c2 = st.columns(2)

            with col_c1:
                st.write("#### Risk Level Distribution")
                risk_counts = results_df['Risk Level'].value_counts()
                labels = [str(x) for x in risk_counts.index.tolist()]
                values = [int(x) for x in risk_counts.values.tolist()]
                color_map = {'HIGH': '#e74c3c', 'MEDIUM': '#f39c12', 'LOW': '#2ecc71'}
                colors_list = [color_map.get(l, '#999999') for l in labels]
                fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
                ax_pie.pie(values, labels=labels, colors=colors_list,
                           autopct='%1.1f%%', startangle=90)
                ax_pie.set_title('Risk Distribution')
                plt.tight_layout()
                st.pyplot(fig_pie, use_container_width=False)
                plt.close(fig_pie)

            with col_c2:
                st.write("#### Churn Probability by Contract")
                fig_bar = px.bar(
                    results_df,
                    x='Customer #',
                    y='Churn Probability (%)',
                    color='Risk Level',
                    color_discrete_map={
                        'HIGH': '#e74c3c',
                        'MEDIUM': '#f39c12',
                        'LOW': '#2ecc71'
                    },
                    title="Churn Probability per Customer"
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            col_c3, col_c4 = st.columns(2)

            with col_c3:
                st.write("#### Contract Type Distribution")
                contract_counts = results_df['Contract'].value_counts()
                fig_contract, ax_c = plt.subplots(figsize=(4, 4))
                ax_c.pie(
                    contract_counts.values,
                    labels=contract_counts.index,
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=['#e74c3c', '#f39c12', '#2ecc71']
                )
                ax_c.set_title('Contract Type Mix')
                plt.tight_layout()
                st.pyplot(fig_contract, use_container_width=False)
                plt.close(fig_contract)

            with col_c4:
                st.write("#### Tenure vs Churn Probability")
                fig_scatter = px.scatter(
                    results_df,
                    x='Tenure (months)',
                    y='Churn Probability (%)',
                    color='Risk Level',
                    color_discrete_map={
                        'HIGH': '#e74c3c',
                        'MEDIUM': '#f39c12',
                        'LOW': '#2ecc71'
                    },
                    title="Tenure vs Churn Probability",
                    size='Monthly Charges ($)'
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

            # ── SECTION 4: Revenue Impact ──
            st.write("---")
            st.write("### 💰 Revenue Impact Analysis")

            avg_monthly = results_df['Monthly Charges ($)'].mean()
            high_risk_revenue = results_df[results_df['Risk Level'] == 'HIGH']['Monthly Charges ($)'].sum()
            total_revenue_at_risk = high_risk_revenue * 12

            col_r1, col_r2, col_r3 = st.columns(3)
            col_r1.metric("Avg Monthly Charges", f"${avg_monthly:.2f}")
            col_r2.metric("Monthly Revenue at Risk", f"${high_risk_revenue:,.2f}")
            col_r3.metric("Annual Revenue at Risk", f"${total_revenue_at_risk:,.2f}")

            st.write("---")
            st.write("### 💡 ROI of Retaining High Risk Customers")

            retention_cost_per_customer = 50
            total_retention_cost = high_risk * retention_cost_per_customer
            revenue_saved = high_risk_revenue * 12
            net_roi = revenue_saved - total_retention_cost

            col_roi1, col_roi2, col_roi3 = st.columns(3)
            col_roi1.metric("Retention Cost", f"${total_retention_cost:,.2f}")
            col_roi2.metric("Revenue Saved (Annual)", f"${revenue_saved:,.2f}")
            col_roi3.metric("Net Benefit", f"${net_roi:,.2f}")

            if net_roi > 0:
                st.success(f"✅ Retaining all HIGH RISK customers saves ${net_roi:,.2f} annually!")
            else:
                st.warning("⚠️ Review retention strategy!")

            # ── SECTION 5: Recommendations Summary ──
            st.write("---")
            st.write("### 💡 Action Plan")

            if high_risk > 0:
                st.error(f"🔴 **{high_risk} HIGH RISK customers** → Call immediately and offer 20% discount!")
            if medium_risk > 0:
                st.warning(f"🟡 **{medium_risk} MEDIUM RISK customers** → Send retention email and free upgrade!")
            if low_risk > 0:
                st.success(f"🟢 **{low_risk} LOW RISK customers** → Send thank you message and referral bonus!")

            # ── SECTION 6: Download Options ──
            st.write("---")
            st.write("### 📥 Download Results")

            col_d1, col_d2 = st.columns(2)

            with col_d1:
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=results_df.to_csv(index=False),
                    file_name="batch_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with col_d2:
                batch_report = f"""
BATCH CHURN PREDICTION REPORT
================================
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Total Customers: {total}

SUMMARY
================================
High Risk   : {high_risk} customers
Medium Risk : {medium_risk} customers
Low Risk    : {low_risk} customers

REVENUE IMPACT
================================
Monthly Revenue at Risk  : ${high_risk_revenue:,.2f}
Annual Revenue at Risk   : ${total_revenue_at_risk:,.2f}
Retention Cost           : ${total_retention_cost:,.2f}
Net Benefit if Retained  : ${net_roi:,.2f}

ACTION PLAN
================================
HIGH RISK   → Call immediately + 20% discount
MEDIUM RISK → Send retention email + Free upgrade
LOW RISK    → Thank you message + Referral bonus
"""
                st.download_button(
                    label="📥 Download Full Report as TXT",
                    data=batch_report,
                    file_name="batch_report.txt",
                    mime="text/plain",
                    use_container_width=True
                )

# ════════════════════════════════
# 🗄️ PREDICTION HISTORY PAGE
# ════════════════════════════════
elif page == "🗄️ Prediction History":
    st.title("🗄️ Prediction History")
    st.write("All predictions made using this app are saved here!")
    st.write("---")

    history_df = load_history()

    if len(history_df) == 0:
        st.warning("No predictions made yet! Go to Predict Churn page first.")
    else:
        st.write("### 📊 Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Predictions", len(history_df))
        col2.metric("High Risk", len(history_df[history_df['risk_level'] == 'HIGH']))
        col3.metric("Medium Risk", len(history_df[history_df['risk_level'] == 'MEDIUM']))
        col4.metric("Low Risk", len(history_df[history_df['risk_level'] == 'LOW']))

        st.write("---")

        st.write("### 🔍 Filter Predictions")
        risk_filter = st.selectbox(
            "Filter by Risk Level",
            ["All", "HIGH", "MEDIUM", "LOW"]
        )

        if risk_filter != "All":
            filtered_df = history_df[history_df['risk_level'] == risk_filter]
        else:
            filtered_df = history_df

        st.write(f"### 📋 Prediction Records ({len(filtered_df)} records)")
        st.dataframe(filtered_df, use_container_width=True)

        st.write("---")

        col_l = st.columns(1)[0]

        with col_l:
            st.write("### Risk Level Distribution")
            risk_counts = history_df['risk_level'].value_counts()
            labels = [str(x) for x in risk_counts.index.tolist()]
            values = [int(x) for x in risk_counts.values.tolist()]
            color_map = {'HIGH': '#e74c3c', 'MEDIUM': '#f39c12', 'LOW': '#2ecc71'}
            colors_list = [color_map.get(l, '#999999') for l in labels]
            fig_pie, ax_pie = plt.subplots(figsize=(3, 3))
            ax_pie.pie(
                values,
                labels=labels,
                colors=colors_list,
                autopct='%1.1f%%',
                startangle=90
            )
            ax_pie.set_title('Risk Distribution')
            plt.tight_layout()
            st.pyplot(fig_pie, use_container_width=False)
            plt.close(fig_pie)

        st.write("---")

        st.download_button(
            label="📥 Download History as CSV",
            data=history_df.to_csv(index=False),
            file_name="prediction_history.csv",
            mime="text/csv"
        )

        st.write("---")
        if st.button("🗑️ Clear All History", type="secondary"):
            delete_all()
            st.success("History cleared successfully!")
            st.rerun()

# ════════════════════════════════
# 💰 CHURN COST CALCULATOR PAGE
# ════════════════════════════════
elif page == "💰 Churn Cost Calculator":
    st.title("💰 Churn Cost Calculator")
    st.write("Calculate how much revenue your company loses due to customer churn!")
    st.write("---")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### 📊 Enter Your Business Numbers")
        total_customers = st.number_input(
            "Total Number of Customers",
            min_value=100, max_value=1000000, value=7043)
        avg_monthly_revenue = st.number_input(
            "Average Monthly Revenue per Customer ($)",
            min_value=1.0, max_value=1000.0, value=65.0)
        churn_rate = st.slider(
            "Current Churn Rate (%)",
            min_value=1, max_value=50, value=26)
        avg_retention_cost = st.number_input(
            "Cost to Retain One Customer ($)",
            min_value=1.0, max_value=500.0, value=50.0)

    with col2:
        st.write("### 🎯 After ML Model Implementation")
        predicted_churn_reduction = st.slider(
            "Expected Churn Reduction with ML (%)",
            min_value=1, max_value=50, value=20)
        model_implementation_cost = st.number_input(
            "ML Model Implementation Cost ($)",
            min_value=0.0, max_value=100000.0, value=5000.0)

    st.write("---")

    if st.button("💰 Calculate Revenue Impact",
                 type="primary", use_container_width=True):

        churned_customers = int(total_customers * churn_rate / 100)
        monthly_revenue_lost = churned_customers * avg_monthly_revenue
        annual_revenue_lost = monthly_revenue_lost * 12
        retention_cost = churned_customers * avg_retention_cost

        reduced_churners = int(churned_customers * predicted_churn_reduction / 100)
        saved_customers = reduced_churners
        monthly_revenue_saved = saved_customers * avg_monthly_revenue
        annual_revenue_saved = monthly_revenue_saved * 12
        net_benefit = annual_revenue_saved - model_implementation_cost
        roi = (net_benefit / model_implementation_cost) * 100 if model_implementation_cost > 0 else 0

        st.write("## 📊 Results")

        st.write("### ❌ Current Situation (Without ML)")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Customers Churning", f"{churned_customers:,}")
        col_b.metric("Monthly Revenue Lost", f"${monthly_revenue_lost:,.2f}")
        col_c.metric("Annual Revenue Lost", f"${annual_revenue_lost:,.2f}")

        st.write("---")

        st.write("### ✅ After ML Model Implementation")
        col_d, col_e, col_f = st.columns(3)
        col_d.metric("Customers Saved", f"{saved_customers:,}")
        col_e.metric("Monthly Revenue Saved", f"${monthly_revenue_saved:,.2f}")
        col_f.metric("Annual Revenue Saved", f"${annual_revenue_saved:,.2f}")

        st.write("---")

        st.write("### 💡 Return on Investment (ROI)")
        col_g, col_h, col_i = st.columns(3)
        col_g.metric("Implementation Cost", f"${model_implementation_cost:,.2f}")
        col_h.metric("Net Annual Benefit", f"${net_benefit:,.2f}")
        col_i.metric("ROI", f"{roi:.1f}%")

        st.write("---")

        if net_benefit > 0:
            st.success(f"""
            ✅ PROFITABLE INVESTMENT!
            By implementing this ML model:
            → You save {saved_customers:,} customers per year
            → You recover ${annual_revenue_saved:,.2f} in annual revenue
            → Your ROI is {roi:.1f}%
            → For every $1 spent, you get ${(annual_revenue_saved/model_implementation_cost):.2f} back!
            """)
        else:
            st.warning(f"""
            ⚠️ Review your numbers!
            Current projection shows negative ROI.
            Try increasing churn reduction % or
            reducing implementation cost.
            """)

        st.write("---")
        calc_report = f"""
CHURN COST CALCULATOR REPORT
==============================
BUSINESS INPUTS
Total Customers          : {total_customers:,}
Avg Monthly Revenue      : ${avg_monthly_revenue}
Current Churn Rate       : {churn_rate}%
Retention Cost           : ${avg_retention_cost}

CURRENT SITUATION
Churning Customers       : {churned_customers:,}
Monthly Revenue Lost     : ${monthly_revenue_lost:,.2f}
Annual Revenue Lost      : ${annual_revenue_lost:,.2f}

AFTER ML MODEL
Customers Saved          : {saved_customers:,}
Monthly Revenue Saved    : ${monthly_revenue_saved:,.2f}
Annual Revenue Saved     : ${annual_revenue_saved:,.2f}

ROI ANALYSIS
Implementation Cost      : ${model_implementation_cost:,.2f}
Net Annual Benefit       : ${net_benefit:,.2f}
ROI                      : {roi:.1f}%
"""
        st.download_button(
            label="📥 Download Calculator Report",
            data=calc_report,
            file_name="churn_cost_report.txt",
            mime="text/plain"
        )