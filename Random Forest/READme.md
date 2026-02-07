# Random Forest Regression – Gym Crowdedness Prediction

## Overview
This project applies a Random Forest Regression model to predict gym crowdedness levels based on time-related features.  
The goal is to solve a real-world regression problem using an ensemble learning method on structured tabular data.

---

## Dataset
The dataset contains records related to gym usage over time.

### Features
- Time-based features extracted from datetime information
- Contextual variables related to gym activity

### Target
- Gym crowdedness level (continuous value)

---

## Project Workflow

### 1) Data Understanding and Feature Inspection
- Loaded the dataset and reviewed its structure
- Converted datetime values into numerical features
- Checked data consistency and basic statistics

---

### 2) Exploratory Data Analysis (EDA)
- Analyzed the distribution of the target variable
- Observed changes in gym crowdedness over time
- Identified peak and low activity periods

---

### 3) Modeling
- Trained a Random Forest Regressor as the baseline model
- Used ensemble trees to capture nonlinear relationships
- Evaluated performance using regression error metrics

---

### 4) Hyperparameter Tuning
- Applied GridSearchCV to tune key Random Forest parameters
- Compared tuned model performance with the baseline model

---

## Results and Notes
- Random Forest performed well on time-based features without heavy feature engineering
- Hyperparameter tuning improved performance compared to default settings
- This project demonstrates the use of ensemble models for practical regression tasks
------------------------------------------------------------------------------------

