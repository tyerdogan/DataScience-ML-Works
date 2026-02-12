# Car Price Prediction with AdaBoost Regressor

This notebook builds a regression pipeline to predict **selling_price** from the CarDekho-style car dataset.
It includes basic cleaning, feature encoding, and model tuning using **AdaBoostRegressor**.

## Workflow
1. Load dataset (CSV)
2. Basic cleaning
   - Drop unnecessary index column
   - Remove duplicates
   - Fix invalid `seats` values
3. Outlier filtering (heuristic thresholds)
   - `selling_price < 10,000,000`
   - `km_driven < 600,000`
4. Train/Test split (80/20)
5. Feature encoding
   - One-hot encoding: `seller_type`, `fuel_type`, `transmission_type`
   - Frequency encoding: `car_name`, `brand`, `model` (learned from train only)
6. Modeling
   - Baseline AdaBoostRegressor
   - Hyperparameter search with RandomizedSearchCV
7. Evaluation
   - R2 score
   - Mean Squared Error (MSE)
   - Mean Absolute Error (MAE)
