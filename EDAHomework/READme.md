## Overview
This repo contains my **Exploratory Data Analysis (EDA)** work on an insurance premium dataset.  
I focused on:
- fixing data types,
- handling missing values with clear rules,
- visualizing distributions/outliers,
- checking relationships between features and **Premium Amount**.

## Dataset
- Kaggle: **Insurance Premium Prediction**
- Link used in notebook: https://www.kaggle.com/datasets/schran/insurance-premium-prediction?resource=download
- 
###  Missing value handling (explicit decisions)
Rules documented in the notebook:
- Numerical:
  - if skewed / outlier-heavy → **median**
- Categorical:
  - use **mode** or a dedicated category (e.g., `"Unknown"`)

### EDA (plots + relationship checks)
Target distribution:
- Histogram of **Premium Amount**
- Calculated **mean / median / skewness**
- Notebook note: premiums are heavily concentrated in the **0–1000** range

The dataset appears **synthetic**:
  - feature values look realistic and include missingness/skew (good for practice),
  - but **there may be little/no true relationship** between features and the target (**Premium Amount**) (I am opened for disccusion)
