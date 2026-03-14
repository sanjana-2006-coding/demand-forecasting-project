import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px

# load trained model
model = joblib.load("sales_forecast_model.pkl")

# page title
st.title("Retail Demand Forecasting System")

st.write("Predict daily demand for a store item using machine learning.")

# dashboard layout
col1, col2 = st.columns(2)

with col1:
    store = st.number_input("Store ID", min_value=1, max_value=10, value=1)
    item = st.number_input("Item ID", min_value=1, max_value=50, value=1)

    year = st.number_input("Year", value=2024)
    month = st.number_input("Month", min_value=1, max_value=12, value=1)
    day = st.number_input("Day", min_value=1, max_value=31, value=1)

with col2:
    dayofweek = st.number_input("Day of Week (0=Mon, 6=Sun)", min_value=0, max_value=6, value=0)

    lag_7 = st.number_input("Sales 7 days ago", value=20.0)
    lag_14 = st.number_input("Sales 14 days ago", value=20.0)
    lag_30 = st.number_input("Sales 30 days ago", value=20.0)

    rolling_mean_7 = st.number_input("7-Day Average Sales", value=20.0)
    rolling_mean_30 = st.number_input("30-Day Average Sales", value=20.0)

# prediction button
if st.button("Predict Demand"):

    features = np.array([[
        store,
        item,
        year,
        month,
        day,
        dayofweek,
        lag_7,
        lag_14,
        lag_30,
        rolling_mean_7,
        rolling_mean_30
    ]])

    prediction = model.predict(features)

    # display prediction
    st.subheader("Prediction Result")
    st.metric("Predicted Sales Demand", round(prediction[0], 2))

    # create trend data
    sales_history = pd.DataFrame({
        "Day": ["30 Days Ago", "14 Days Ago", "7 Days Ago", "Predicted Today"],
        "Sales": [rolling_mean_30, lag_14, lag_7, prediction[0]]
    })

    # create demand trend chart
    fig = px.line(
        sales_history,
        x="Day",
        y="Sales",
        markers=True,
        title="Demand Trend"
    )

    st.plotly_chart(fig, use_container_width=True)

# sidebar
st.sidebar.title("Project Information")

st.sidebar.write("""
**Model Used**
XGBoost Regressor

**Feature Engineering**
- Lag Features (7, 14, 30 days)
- Rolling Mean Features
- Calendar Features

**Evaluation Metrics**
MAE ≈ 2.59  
RMSE ≈ 3.23
""")

st.sidebar.write("""
**Project Description**

This application predicts retail item demand based on historical sales patterns.  
The model was trained using machine learning and time-series feature engineering.
""")