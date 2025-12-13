# End-to-End Time Series Sales Forecasting with Python (StatsForecast)

Time series forecasting is one of the most practical applications of data science—especially in **sales and demand prediction**. In this blog, we’ll walk through an **end-to-end time series forecasting pipeline** using Python, where we predict future sales from historical data using **classical statistical models** powered by the `statsforecast` library.

This article is based on a complete, working notebook and is written to be **beginner-friendly**, while still covering industry-grade practices such as:
- Data preprocessing for panel time series
- Baseline forecasting models
- Model evaluation
- Advanced statistical models
- Visual diagnostics

---

## 1. Problem Statement

We are given **daily sales data** for multiple products (e.g., *BAGUETTE*, *CROISSANT*). Each product forms its own time series. Our goal is to:

> **Forecast the next 7 days of sales for each product using historical data.**

This is a **panel (multi-series) time series forecasting** problem.

---

## 2. Libraries and Environment Setup

We begin by installing and importing the required libraries:

- **NumPy & Pandas** → numerical and tabular data handling
- **StatsForecast** → fast and scalable statistical forecasting
- **UtilsForecast** → visualization utilities

```python
!pip install statsforecast utilsforecast
```

```python
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from utilsforecast.plotting import plot_series
```

Why `StatsForecast`?
- Extremely fast (written in Numba)
- Designed for **large-scale forecasting**
- Supports multiple models in parallel

---

## 3. Loading the Sales Dataset

The dataset is loaded directly from a public GitHub repository:

```python
df = pd.read_csv("https://raw.githubusercontent.com/marcopeix/youtube_tutorials/main/data/sales.csv")
```

### Dataset Structure

| Column | Description |
|------|------------|
| unique_id | Product name / SKU |
| ds | Date (daily frequency) |
| y | Sales quantity |
| unit_price | Price (removed later) |

This format follows the **standard StatsForecast convention**:
- `unique_id` → series identifier
- `ds` → timestamp
- `y` → target variable

---

## 4. Data Cleaning and Filtering

### 4.1 Ensuring Sufficient History

Forecasting models require enough historical data. We remove products with fewer than **28 observations**:

```python
df = df.groupby("unique_id").filter(lambda x: len(x) >= 28)
```

This step prevents unstable or misleading forecasts.

### 4.2 Removing Unnecessary Features

Since we are doing **univariate forecasting**, we drop `unit_price`:

```python
df = df.drop(["unit_price"], axis=1)
```

---

## 5. Exploratory Time Series Visualization

Before modeling, it’s critical to **visualize trends and seasonality**.

```python
plot_series(df=df, ids=["BAGUETTE", "CROISSANT"], palette="viridis")
```

### Key Insights:
- Clear daily fluctuations
- Different sales patterns per product
- Evidence that **per-series modeling** is appropriate

We also visualize a limited window to focus on recent behavior:

```python
plot_series(df=df, ids=["BAGUETTE", "CROISSANT"], max_insample_length=56)
```

---

## 6. Baseline Forecasting Models

Baseline models are essential—they give us a **minimum performance benchmark**.

### Models Used

- **Naive** – repeats the last observed value
- **Seasonal Naive** – repeats last season’s value
- **Historic Average** – mean of past values

```python
from statsforecast.models import Naive, SeasonalNaive, HistoricAverage

models = [
    Naive(),
    SeasonalNaive(season_length=7),
    HistoricAverage()
]
```

We forecast the next **7 days**:

```python
horizon = 7
sf = StatsForecast(models=models, freq="D")
preds = sf.forecast(df=df, h=horizon)
```

---

## 7. Visualizing Baseline Predictions

```python
plot_series(
    df=df,
    forecasts_df=preds,
    ids=["BAGUETTE"],
    max_insample_length=30
)
```

This helps us visually compare:
- Actual historical values
- Forecasted trajectories

---

## 8. Model Evaluation Strategy

We evaluate models using a **holdout test set**.

### Train–Test Split

- Last **7 days** → test set
- Remaining data → training set

```python
test = df.groupby("unique_id").tail(7)
train = df.drop(test.index)
```

### Error Metric

We use **Mean Absolute Error (MAE)**:

```python
from utilsforecast.losses import mae
```

MAE is:
- Easy to interpret
- Robust to outliers
- Suitable for sales data

---

## 9. Advanced Statistical Models

After baselines, we introduce more powerful models.

### 9.1 AutoARIMA

Automatically finds the best ARIMA configuration:

```python
from statsforecast.models import AutoARIMA
```

Strengths:
- Captures trend & autocorrelation
- Minimal manual tuning

### 9.2 Exponential Smoothing (ETS)

```python
from statsforecast.models import ETS
```

Strengths:
- Excellent for level + trend + seasonality
- Interpretable components

---

## 10. Training and Forecasting with Advanced Models

```python
models = [
    AutoARIMA(season_length=7),
    ETS(season_length=7)
]

sf = StatsForecast(models=models, freq="D")
preds = sf.forecast(df=train, h=7)
```

---

## 11. Quantitative Evaluation

```python
mae_df = mae(test, preds, id_col="unique_id", time_col="ds", target_col="y")
```

This allows us to:
- Rank models objectively
- Choose the best-performing approach per product

---

## 12. Key Takeaways

### What We Learned

- Panel time series forecasting is straightforward with StatsForecast
- Baseline models are **non-negotiable**
- Statistical models still perform extremely well for sales data
- Visualization is just as important as metrics

### When to Use This Approach

✅ Small–medium datasets  
✅ Limited compute resources  
✅ Need for fast & interpretable forecasts

---

## 13. What’s Next?

Possible extensions:
- Add **promotions & prices** as exogenous variables
- Compare with **ML models (XGBoost, LightGBM)**
- Deploy forecasts via **API or dashboard**

---

## Final Thoughts

Even in the era of deep learning, **classical time series models remain incredibly powerful**—especially when used correctly.

If you’re building real-world forecasting systems, mastering tools like `StatsForecast` is a huge advantage.

Happy Forecasting 📈

---

*If you found this useful, feel free to share it or adapt it for your own datasets.*

