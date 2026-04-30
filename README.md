
# Telco Customer Churn Analysis

## Business Problem
A telecom company is losing 26.5% of its customers annually,
representing $1.67M in revenue at risk. This project identifies
the key drivers of churn and builds a prediction model to flag
at-risk customers before they leave.

## Dataset
- Source: IBM Watson Telco Customer Churn Dataset
- 7,043 customers, 21 features

## Key Findings
- New customers (0–6 months) churn at 53.3%
- Month-to-month contracts churn at 42.7% vs 2.8% for 2-year contracts
- Fiber Optic customers churn at 41.9% despite paying premium prices
- Customers with Tech Support churn at 15% vs 42% without

## Approach
1. Exploratory Data Analysis
2. Data Preprocessing & Encoding
3. Random Forest Classification Model
4. Confusion Matrix & Business Impact Analysis

## Model Performance
- Accuracy  : 79.6%
- Classification_Report:
<img width="718" height="210" alt="Screenshot 2026-04-30 at 18-01-27 Converting raw data into client-ready business narratives - Claude" src="https://github.com/user-attachments/assets/d6989054-dc1c-4b45-8646-2e4f452b9f3c" />


## Tools Used
Python, Pandas, Scikit-learn, Matplotlib, Seaborn

## Future Improvements
- Threshold tuning to improve churn recall
- Handle class imbalance with SMOTE
- Test XGBoost for performance comparison
