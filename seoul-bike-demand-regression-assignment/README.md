# Seoul Bike Demand Regression (EDA + Linear Regression)

This mini-project is a course assignment (Linear Regression) notebook that demonstrates a complete regression workflow on an hourly bike demand dataset:
EDA → feature engineering → preprocessing → baseline regression → cross-validation  → regularization → sanity-check scenario test.

> Developed and executed in **Google Colab**.

---

## Dataset
Hourly aggregated bike rental demand data from Seoul’s public bike-sharing system, combined with ASOS weather variables (temperature, precipitation, wind speed, humidity, solar radiation, etc.).

- Target: `Rented_Bike_Count`
- More dataset + licensing details: see **`DATASET.md`**

---

## Methodology 
- **EDA & sanity checks:** inspect schema, missing values, basic distributions.
- **Feature engineering:** parse `Date` and extract calendar features.
- **Train/test split** and define `X` / `y`.
- **Preprocessing (leakage-safe):** `ColumnTransformer` with scaling (numeric) + one-hot encoding (categorical), fit on train only.
- **Modeling:** baseline Linear Regression + regularization (Ridge/Lasso/ElasticNet).
- **Evaluation:** metrics on original scale (R² / MAE / RMSE) with log-target inversion.
- **Validation:** K-Fold CV using `Pipeline(preprocessor, model)` to avoid leakage within folds.


