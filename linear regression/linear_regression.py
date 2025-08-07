import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("Housing.csv")

# -----------------------------
#  1. Preprocessing
# -----------------------------
X = df.drop(columns="price")
y = df["price"]

# Separate columns by type
categorical_cols = X.select_dtypes(include='object').columns.tolist()
numerical_cols = X.select_dtypes(exclude='object').columns.tolist()

# Encode categorical columns
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(drop='first'), categorical_cols)],
    remainder='passthrough'
)

X_processed = preprocessor.fit_transform(X)

# -----------------------------
#  2. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# -----------------------------
#  3. Train Multiple Linear Regression
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# -----------------------------
#  4. Evaluate the Model
# -----------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"MAE: ₹{mae:.2f}")
print(f"MSE: ₹{mse:.2f}")
print(f"R² Score: {r2:.4f}")

# -----------------------------
#  5. Coefficient Interpretation
# -----------------------------
feature_names = preprocessor.get_feature_names_out()
coefficients = model.coef_
intercept = model.intercept_

print("\n--- Model Coefficients ---")
print(f"Intercept: ₹{intercept:.2f}")
for name, coef in zip(feature_names, coefficients):
    print(f"{name}: {coef:.2f}")

# -----------------------------
#  6. Simple Linear Regression (area vs price)
# -----------------------------
X_simple = df[['area']]
y_simple = df['price']

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_simple, y_simple, test_size=0.2, random_state=42)

model_simple = LinearRegression()
model_simple.fit(X_train_s, y_train_s)
y_pred_s = model_simple.predict(X_test_s)

# Plotting
plt.figure(figsize=(8, 5))
plt.scatter(X_test_s, y_test_s, color='blue', label='Actual Price')
plt.plot(X_test_s, y_pred_s, color='red', linewidth=2, label='Regression Line')
plt.xlabel('Area (sq ft)')
plt.ylabel('Price')
plt.title('Simple Linear Regression: Area vs Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print coefficient
print("\n--- Simple Linear Regression ---")
print(f"Intercept: ₹{model_simple.intercept_:.2f}")
print(f"Coefficient for area: ₹{model_simple.coef_[0]:.2f}")

