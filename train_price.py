
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================
# 1. CREATE DATASET
# ==========================================

data = {
    "area": [
        50, 60, 70, 80, 90, 100, 110, 120,
        130, 140, 150, 160, 170, 180, 190, 200
    ],
    "bedrooms": [
        1, 1, 2, 2, 2, 3, 3, 3,
        3, 4, 4, 4, 4, 4, 5, 5
    ],
    "age": [
        20, 15, 25, 10, 18, 12, 8, 20,
        9, 15, 7, 5, 11, 4, 6, 2
    ],
    "price": [
        450000, 560000, 640000, 820000,
        880000, 990000, 1150000, 1190000,
        1330000, 1400000, 1550000, 1700000,
        1760000, 1880000, 1990000, 2150000
    ]
}

df = pd.DataFrame(data)


# ==========================================
# 2. FEATURES AND TARGET
# ==========================================

X = df.drop(columns=["price"])
y = df["price"]


# ==========================================
# 3. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# 4. TRAIN BASELINE MODEL
# ==========================================

baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_test)

baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions
)


# ==========================================
# 5. TRAIN LINEAR REGRESSION MODEL
# ==========================================

model = LinearRegression()
model.fit(X_train, y_train)


# ==========================================
# 6. EVALUATE MODEL
# ==========================================

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)


print("\n===== MODEL COMPARISON =====")
print(f"Baseline MAE: {baseline_mae:,.2f} MAD")
print(f"Model MAE:    {mae:,.2f} MAD")

print("\n===== MODEL EVALUATION =====")
print(f"MAE:  {mae:,.2f}")
print(f"MSE:  {mse:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"R2:   {r2:.4f}")


# ==========================================
# 7. PREDICT NEW HOUSE PRICE
# ==========================================

new_house = pd.DataFrame({
    "area": [125],
    "bedrooms": [3],
    "age": [7]
})

predicted_price = model.predict(new_house)[0]

print("\n===== NEW PREDICTION =====")
print(f"Predicted house price: {predicted_price:,.0f} MAD")


# ==========================================
# 8. SAVE TRAINED MODEL
# ==========================================

project_dir = Path(__file__).resolve().parent

models_dir = project_dir / "models"
models_dir.mkdir(parents=True, exist_ok=True)

model_path = models_dir / "price_model.joblib"

joblib.dump(model, model_path)

print(f"\nModel saved to: {model_path}")


# ==========================================
# 9. SAVE EVALUATION METRICS
# ==========================================

metrics = {
    "model": "LinearRegression",
    "mae": float(mae),
    "mse": float(mse),
    "rmse": float(rmse),
    "r2": float(r2),
    "baseline_mae": float(baseline_mae)
}

reports_dir = project_dir / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)

metrics_path = reports_dir / "metrics.json"

with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

print(f"Metrics saved to: {metrics_path}")
print("\nPipeline completed successfully!")
