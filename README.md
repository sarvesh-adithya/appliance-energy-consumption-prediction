# Predicting Appliance Energy Consumption Using Machine Learning

A machine learning project to predict household appliance energy consumption
from ambient sensor readings (temperature, humidity, time), with a strong
emphasis on rigorous time-series evaluation and honest reporting of model
limitations.

## Dataset

- **Source**: UCI Appliances Energy Prediction dataset
- **Size**: 19,735 rows, 29 columns, 10-minute interval readings from a single household
- **Features**: 9 indoor temperature sensors (T1–T9), 9 indoor humidity sensors
  (RH_1–RH_9), outdoor weather (temperature, humidity, wind speed, visibility,
  pressure), lights energy usage, and two intentionally-included random noise
  columns (rv1, rv2) used to sanity-check feature selection
- **Target**: `Appliances` — energy consumption in Wh

## Feature Engineering

- Extracted `hour`, `day`, `month`, `weekday`, `is_weekend` from the timestamp
- Computed `avg_temp` and `avg_humidity` across all room sensors
- Added `rolling_avg_appliances_24h` — a rolling average of the household's
  own recent energy usage over the past 24 hours, computed with `.shift(1)`
  to avoid leaking the current row's value into its own feature

## Methodology

Five regression models were compared: Linear Regression, Decision Tree,
Random Forest, Gradient Boosting, and XGBoost.

### A note on evaluation methodology (important)

An earlier version of this analysis used a **random 80/20 train/test split**,
which produced an optimistic Random Forest R² of **0.59**. Because this
dataset is a correlated time series (10-minute intervals), a random split
allows training rows to sit just minutes away from test rows — which are
nearly identical in conditions. This is **temporal data leakage**, and it
made the original evaluation unrealistically favorable.

This was corrected using a **chronological split** — training on the earlier
~80% of the timeline and testing on the later ~20%, which simulates how the
model would actually be used in deployment: predicting a genuine future
period it has never seen anything close to.

## Model Performance (Corrected, Chronological Split)

| Model | R² | Notes |
|---|---|---|
| Naive baseline (24h rolling average, no model) | 0.019 | Performance floor |
| Random Forest (regularized) | -0.658 | Fails to extrapolate to unseen future conditions |
| Decision Tree | -5.34 | Same extrapolation failure |
| Gradient Boosting | -10.52 | Same extrapolation failure |
| XGBoost | -7.86 | Same extrapolation failure |
| **Linear Regression** | **0.120** | Best honest result |

### Why every tree-based model failed here

Tree-based models (Random Forest, Decision Tree, Gradient Boosting, XGBoost)
predict by averaging training examples within a leaf node — they **cannot
output values outside the range they saw during training.** Regularizing
Random Forest (capping tree depth, raising minimum leaf size) improved its
score substantially (from roughly -2.8 to -0.66) but did not make it positive,
which indicates the issue isn't only overfitting — it's genuine
**non-stationarity**: the relationship between ambient conditions and
appliance usage shifts over time in ways these features don't fully capture.

Linear Regression, as a formula-based model, can extrapolate beyond the
training value range (even if imperfectly), which is why it remains the more
defensible choice despite the tree-based models winning on the original,
leakage-inflated random split.

### Takeaway

Ambient sensor data (temperature, humidity, time) has a relatively low
ceiling for predicting appliance-level energy usage in this household. The
real driver of appliance usage is human behavior, which these features only
partially capture. A production system aiming to substantially improve on
this would likely need occupancy sensors or appliance-level usage logs, and
ideally a scheduled retraining pipeline to keep the model current as
household patterns shift over time.

## Interpretability

SHAP (TreeExplainer) was used on the Random Forest model to identify the
top drivers of predicted energy consumption, including hour of day, lights
usage, recent rolling energy usage, and several indoor temperature/humidity
sensors.

## Tech Stack

- **Data & Modeling**: Python, Pandas, NumPy, Scikit-learn, XGBoost
- **Interpretability**: SHAP
- **Model persistence**: joblib

## Project Structure

```
├── Predicting Appliance Energy Consumption In Households.ipynb   # Full analysis
├── appliance_energy_model.pkl      # Serialized model
├── requirements.txt                # Dependencies
└── README.md
```

## Key Learnings From This Project

1. Random train/test splits are unsafe for correlated time-series data —
   always use a time-aware split for anything sequential.
2. A model that looks best on a flawed evaluation can be the worst choice in
   practice — always sanity-check strong results against a naive baseline.
3. Tree-based models cannot extrapolate beyond their training value range,
   which matters a great deal when deploying on genuinely new future data.
4. Regularization can fix overfitting, but it cannot fix non-stationarity —
   these are two separate problems that require different solutions.
