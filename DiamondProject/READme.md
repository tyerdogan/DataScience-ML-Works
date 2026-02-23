# 💎 Diamond Price Predictor

A web application that predicts diamond prices based on physical and quality characteristics using a pre-trained machine learning model.

---

## Overview

Users can enter diamond features through a clean web interface and instantly receive an estimated market price. The model was trained with scikit-learn and is served via a FastAPI backend.

---

## Tech Stack

| Layer     | Technology                  |
|-----------|-----------------------------|
| Backend   | FastAPI, Uvicorn             |
| ML Model  | scikit-learn (pre-trained)   |
| Data      | pandas                       |
| Frontend  | Jinja2 Templates, HTML/CSS   |

---

## Features

- Predict diamond price from 9 input features
- Categorical encoding for `cut`, `color`, and `clarity`
- Feature scaling applied before inference
- Responsive single-page UI

---

## Input Features

| Feature   | Type    | Example     |
|-----------|---------|-------------|
| `carat`   | float   | 0.50        |
| `cut`     | string  | Ideal       |
| `color`   | string  | E           |
| `clarity` | string  | VS1         |
| `depth`   | float   | 61.5        |
| `table`   | float   | 55.0        |
| `x`       | float   | 4.95        |
| `y`       | float   | 4.98        |
| `z`       | float   | 3.07        |

---

## Getting Started

### 1. Clone the repository

```bash
git clone <repository-url>
cd DiamondProject
```

### 2. Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the application

```bash
uvicorn app:app --reload
```

Then open your browser at your localhost

---

## API

### `POST /predict`

**Request body (JSON):**
```json
{
  "carat": 0.50,
  "cut": "Ideal",
  "color": "E",
  "clarity": "VS1",
  "depth": 61.5,
  "table": 55.0,
  "x": 4.95,
  "y": 4.98,
  "z": 3.07
}
```

**Response:**
```json
{
  "predicted_price": 1823.47
}
```

---

## Project Structure

```
DiamondProject/
├── app.py                        # FastAPI application
├── model_tests.py                # Model testing script
├── diamonds_model_complete.pkl   # Pre-trained model, encoders & scaler
├── diamonds_test_data_scaled.csv # Scaled test dataset
├── requirements.txt
└── templates/
    └── index.html                # Frontend UI
```

