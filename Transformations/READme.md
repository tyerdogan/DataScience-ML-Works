### 1) House Prices — Model Benchmark + Feature/Target Transformations
**Notebook:** `HousePriceXGBoostRegressorTransformation.ipynb`  
**Data file:** `21-housing.csv`  
**Goal:** Predict `median_house_value` using structured housing features (includes categorical `ocean_proximity`).

**Highlights**
- Quick EDA with distribution plots
- Basic missing value handling (e.g., median imputation)
- Categorical encoding with `pd.get_dummies(..., drop_first=True)`
- Optional experiments with **feature transformations** (e.g., `log1p`, Yeo–Johnson)
- Benchmarking multiple regressors (Linear/Ridge/Lasso/KNN/Tree/RF/AdaBoost/GB/XGBoost)
- Metrics: **MAE, RMSE, R²**

---

### 2) Boston Housing — Yeo–Johnson (X) + Box–Cox (y) vs Baseline Linear Regression
**Notebook:** `BostonHousePriceTransformations.ipynb`  
**Data file:** `23-boston.csv`  
**Goal:** Predict `MEDV` (house value) and observe how transformations affect distributions and baseline model performance.

**Highlights**
- Load Boston dataset with explicit column names
- Apply **Yeo–Johnson** transformation to input features (`X`)
- Apply **Box–Cox** transformation to target (`y`) + inverse transform for predictions
- Compare transformed vs non-transformed **Linear Regression**
- Metrics: **R², MSE**
- Visual comparison of feature distributions (before vs after)

---

### 3) Medical Insurance Cost — LightGBM + RandomizedSearchCV + Box–Cox Target Transform
**Notebook:** `MedicalCostRegressor.ipynb`  
**Data file:** `24-medical_cost.csv`  
**Goal:** Predict `charges` (medical cost).

**Highlights**
- Quick EDA (counts by `sex`, `smoker`, `region`)
- Encode binary fields (`sex`, `smoker`) via mapping
- One-hot encode `region` using `ColumnTransformer + OneHotEncoder`
- Train **LightGBM Regressor**
- Hyperparameter tuning with **RandomizedSearchCV**
- Optional **Box–Cox** target transform with inverse transform for prediction comparison
- Metrics: **R², MSE** (plus tuning score based on RMSE)

---
