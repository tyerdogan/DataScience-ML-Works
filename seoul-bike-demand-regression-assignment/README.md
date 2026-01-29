# Seoul Bike Demand Regression (EDA + Linear Regression)

This mini-project is a course assignment (Linear Regression) notebook that demonstrates a complete regression workflow on an hourly bike demand dataset:
EDA → feature engineering → preprocessing → baseline regression → cross-validation  → regularization → sanity-check scenario test.

Developed and executed in **Google Colab**.

---

## Dataset & License
Hourly aggregated bike rental demand data from Seoul’s public bike-sharing system, combined with ASOS weather variables (temperature, precipitation, wind speed, humidity, solar radiation, etc.).

- Target: `Rented_Bike_Count`

Sources: Seoul Open Data Plaza (data.seoul.go.kr) + KMA ASOS (data.kma.go.kr).  
License: KOGL Type 1 (Attribution Required) — [KOGL License](https://www.kogl.or.kr/info/license.do)
This is an independent educational work; not affiliated with or endorsed by the data providers.


---

## Methodology 
- **EDA & sanity checks:** inspect schema, missing values, basic distributions.
- **Feature engineering:** parse `Date` and extract calendar features.
- **Train/test split** and define `X` / `y`.
- **Preprocessing (leakage-safe):** `ColumnTransformer` with scaling (numeric) + one-hot encoding (categorical).
- **Modeling:** baseline Linear Regression + regularization (Ridge/Lasso/ElasticNet).
- **Evaluation:** metrics on original scale (R² / MAE / RMSE) with log-target inversion.
- **Validation:** K-Fold CV using `Pipeline(preprocessor, model)` to avoid leakage within folds.
- **Note: The data is time-indexed (hourly). This notebook uses random split and K-Fold CV under an i.i.d. assumption; for strict forecasting evaluation, prefer a chronological split or TimeSeriesSplit.**

---

## Acknowledgements
Completed as an assignment from the Udemy course **"Python for Data Science and Machine Learning"** by **Atıl Samancıoğlu**.
Independent educational work; not affiliated with or endorsed by Udemy or the instructor.


