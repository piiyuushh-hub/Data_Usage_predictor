# Telecom Monthly Data Usage Prediction

## Problem Overview
This project predicts monthly internet data usage of telecom customers using machine learning regression models. The aim is to help telecom companies understand customer usage behavior and optimize network planning.

## Dataset
The dataset was synthetically generated to simulate real-world telecom data. It contains noisy values, missing entries, and outliers to reflect realistic scenarios.

## Features Used
- Customer age
- Monthly recharge amount
- Call minutes
- SMS count
- Internet speed
- Roaming usage
- Tenure
- Device type
- Plan type
- Network type
- Region

Non-informative features such as customer ID and payment method were removed during feature selection.

## Machine Learning Models
The following regression models were trained and evaluated:
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Outlier treatment was performed using the IQR method, which significantly improved model performance.

## Model Selection
Linear Regression was selected as the final model based on the highest R² score and lowest error metrics.

## Deployment
The model was deployed locally using Streamlit, allowing users to input customer details and receive predicted monthly data usage along with visual insights.

## How to Run
```bash
pip install pandas numpy scikit-learn streamlit matplotlib seaborn
streamlit run app.py
