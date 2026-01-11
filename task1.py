# task1.py
# House Price Prediction using Linear Regression

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("train.csv")

# -----------------------------
# Select Features & Target
# -----------------------------
features = ["GrLivArea", "BedroomAbvGr", "FullBath"]
target = "SalePrice"

df = data[features + [target]]

# -----------------------------
# Handle Missing Values
# -----------------------------
df = df.fillna(df.median())

X = df[features]
y = df[target]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Train Linear Regression Model
# -----------------------------
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# -----------------------------
# Make Predictions
# -----------------------------
y_pred = model.predict(X_test_scaled)

# -----------------------------
# Model Evaluation
# -----------------------------
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Model Evaluation Results")
print("------------------------")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")

# -----------------------------
# Coefficient Interpretation
# -----------------------------
coefficients = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
})

print("\nFeature Coefficients")
print("--------------------")
print(coefficients)
