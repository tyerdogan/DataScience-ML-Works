# Random Forest Regression – Gym Crowdedness Prediction

## Overview
This project solves a regression problem by predicting gym crowdedness using time-based features.
Multiple regression models are compared, with a focus on Random Forest performance.

## Dataset
The dataset contains gym usage records with datetime information and numerical features.
The target variable is a continuous crowdedness value.

## Workflow
- Converted datetime features into numerical form
- Split data into training and test sets
- Applied feature scaling
- Trained and compared several regression models
- Tuned KNN and Random Forest models using RandomizedSearchCV
- Evaluated models using RMSE, MAE, and R²

## Notes
RandomizedSearchCV was used to efficiently tune model hyperparameters over a large search space.
----------------------------------------------------------
# Random Forest Classification – Income Prediction

## Overview
This project builds a Random Forest classification model to predict income categories from demographic data.

## Dataset
The dataset includes categorical and numerical features related to personal income.
The target variable is a binary income class.

## Workflow
- Performed basic exploratory data analysis
- Split data into training and test sets
- Applied target mean encoding and one-hot encoding
- Trained a Random Forest classifier
- Tuned model hyperparameters using RandomizedSearchCV
- Evaluated performance using standard classification metrics

## Notes
Different encoding strategies were used to handle categorical features before training the model.
