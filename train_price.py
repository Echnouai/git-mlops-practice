
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from pathlib import Path
import joblib
import numpy as np


# 1. Dataset
data = {
    "area": [50, 60, 70, 80, 90, 100, 110, 120,
             130, 140, 150, 160, 170, 180, 190, 200],

    "bedrooms": [1, 1, 2, 2, 2, 3, 3, 3,
                 3, 4, 4, 4, 4, 4, 5, 5],

    "age": [20, 15, 25, 10, 18, 12, 8, 20,
            9, 15, 7, 5, 11, 4, 6, 2],

    "price": [450000, 560000, 640000, 820000,
              880000, 990000, 1150000, 1190000,
              1330000, 1400000, 1550000, 1700000,
              1760000, 1880000, 1990000, 2150000]
}

df = pd.DataFrame(data)

# 2. Features and target
X = df[["area", "bedrooms", "age"]]
y = df["price"]

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 4. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Evaluate model
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("===== MODEL EVALUATION =====")
print(f"MAE:  {mae:,.2f}")
print(f"MSE:  {mse:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"R2:   {r2:.4f}")

# 6. Predict a new house
new_house = pd.DataFrame({
    "area": [125],
    "bedrooms": [3],
    "age": [7]
})

price = model.predict(new_house)[0]
print(f"\nPredicted house price: {price:,.0f} MAD")

# 7. Save trained model
model_path = Path(__file__).resolve().parent / "models" / "price_model.joblib"
model_path.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(model, model_path)

print(f"\nModel saved to: {model_path}")
