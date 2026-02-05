### 1) `QualityOfLifeDataRegressor.ipynb`
**Goal:** Predict **`age_at_death`** using demographic + time-allocation features (work/rest/sleep/exercise) and categorical attributes (gender, occupation).

**Dataset:** Kaggle – “Quality of Life Data”  
(Linked inside the notebook as a reference)

**Core pipeline**
- Drop ID-like columns
- Encode categorical features (label encoding)
- Train/test split
- Feature scaling (for linear/SVR models)
- Train multiple regressors + compare with standard regression metrics

**Models explored**
- Linear Regression
- Ridge Regression (+ `GridSearchCV`)
- Support Vector Regressor (SVR, RBF) (+ `GridSearchCV`)

**Best reported results (from notebook outputs)**
- **SVR (tuned):** `R2 ≈ 0.5723`, `MAE ≈ 6.4229`
- **Ridge (baseline/tuned in notebook):** ~`R2 ≈ 0.28`
- **(Other baselines shown):** up to ~`R2 ≈ 0.546` depending on the model variant

### 2) `CarEvaluationDecisionTreeClassifier.ipynb`
**Goal:** Predict car acceptability class (**`class`**) from categorical attributes.

**Core pipeline**
- Assign column names: `buying, maintenance, doors, persons, lug_boot, safety, class`
- Feature engineering:
  - Convert string categories like doors/persons into numeric-friendly representations
  - Encode categorical inputs
- Train/test split
- Train Decision Tree(s)
- Evaluate with accuracy + confusion matrix + classification report
- Hyperparameter tuning with `GridSearchCV`

**Reported results (from notebook outputs)**
- **Car Evaluation Decision Tree:** `Accuracy ≈ 0.9730` (test set)
- Confusion matrix + per-class precision/recall/F1 printed in the notebook

**Extra section**
- A small **Iris** decision-tree/pruning demo is included:
  - **Iris Decision Tree:** `Accuracy ≈ 0.9333` (as shown in the notebook output)

