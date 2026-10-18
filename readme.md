# House Price Prediction — ML Training Pipeline

A simple machine learning project for predicting house prices using Linear Regression with scikit-learn.

## Features

- House price prediction based on area, bedrooms, and age
- Train/test data splitting
- Baseline comparison using DummyRegressor
- Model evaluation using MAE, MSE, RMSE, and R²
- Model serialization using Joblib
- Evaluation metrics stored in JSON

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Git and GitHub

## Installation

Clone the repository:

```bash
git clone https://github.com/Echnouai/git-mlops-practice.git
cd git-mlops-practice
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Training

Run the training pipeline:

```bash
python train_price.py
```

The script trains the model, evaluates its performance, predicts a sample house price, and saves the outputs.

## Outputs

- `models/price_model.joblib` — serialized trained model
- `reports/metrics.json` — evaluation results

## Dataset

The project uses a small synthetic dataset of 16 houses. It is intended for learning and demonstrating ML workflows, not for real-world property valuation.

## Future Improvements

- Experiment tracking using MLflow
- Prediction API using FastAPI
- Containerization using Docker
- CI/CD using GitHub Actions
- Deployment and monitoring