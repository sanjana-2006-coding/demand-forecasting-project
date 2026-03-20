# 🛒 Retail Demand Forecasting using Machine Learning

## 📌 Overview
A machine learning system that predicts **daily product demand** for retail stores using historical sales data.

The model captures:
- Weekly trends
- Seasonality
- Recent demand patterns

Goal: Improve **inventory decisions** by forecasting demand accurately.

---

## ❗ Problem Statement

Retail businesses face:
- Overstocking → financial loss  
- Understocking → missed sales  

This project predicts:

> **“How many units of a product will be sold on a given day?”**

---

## 📊 Dataset

- Source: Kaggle  
- Dataset: Store Item Demand Forecasting Dataset  
- Link: https://www.kaggle.com/datasets/dhrubangtalukdar/store-item-demand-forecasting-dataset  

**Note:** Dataset is excluded due to size (>100MB)

---

## ⚙️ Feature Engineering

### Calendar Features
- Year, Month, Day
- Day of Week

### Lag Features (Most Important)
- Sales (t-7)
- Sales (t-14)
- Sales (t-30)

➡️ Captures historical demand behavior  

### Rolling Features
- 7-day moving average  
- 30-day moving average  

➡️ Captures recent trends  

---

## 🤖 Models Used

- Linear Regression  
- Random Forest Regressor  
- XGBoost Regressor  

---

## 📈 Model Performance

| Model               | MAE   | RMSE |
|--------------------|------|------|
| Linear Regression  | 3.61 | —    |
| Random Forest      | 2.72 | —    |
| XGBoost            | **2.58** | — |
| Tuned Random Forest| 2.71 | 3.41 |
| Tuned XGBoost      | **2.59** | **3.23** |

### Key Insight
- XGBoost achieved best performance  
- Predictions are within **±2–3 units**  

---

## 🔍 Feature Importance

Top contributors:
- Lag 7  
- Rolling Mean (7-day)  
- Lag 14, Lag 30  

➡️ Recent sales history dominates prediction

---

## 🧰 Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn
- Linear Regression
- Random Forest  
- XGBoost  
- Streamlit  
- Plotly  

---

## 🌐 Web Application

Built using **Streamlit**

### Features
- Input store, item, and past sales  
- Instant demand prediction  
- Interactive visualization  

### Run Locally

```bash
pip install -r src/requirements.txt
streamlit run src/app.py
````

---

## 📁 Project Structure

```
demand-forecasting-project/
│
├── data/                # Ignored (large dataset)
├── notebooks/           # EDA and analysis
├── results/             # outputs and metrics
├── src/
│   ├── app.py           # Streamlit app
│   ├── requirements.txt
│   └── sales_forecast_model.pkl (ignored)
```

---

## 🎯 Key Learnings

* Time-series feature engineering is critical
* Lag features significantly improve accuracy
* Tree-based models outperform linear models
* Hyperparameter tuning gives incremental gains
* Deployment is part of ML, not optional

---

## 🚀 Future Improvements

* Multi-step forecasting (next 7 days)
* Real-time data integration
* Advanced dashboard UI
* Cloud deployment

---

## 🧾 Conclusion

End-to-end ML pipeline:

```
Data → Features → Model → Evaluation → Deployment
```

The system delivers accurate retail demand forecasts and is deployed as an interactive application.
