import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Data
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([2.5, 3.6, 4.1, 5.2, 6.3, 7.0, 8.2, 9.1, 9.9, 11.2])

# Linear Regression Model
model = LinearRegression()
model.fit(X, y)

# Predictions
predictions = model.predict(X)

# Coefficients
coef = model.coef_[0]
intercept = model.intercept_

# Print the model parameters
print(f"Coefficient: {coef}")
print(f"Intercept: {intercept}")
print(f"Predictions: {predictions}")

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(X, y, 'o', label='Original Data', markersize=8)
plt.plot(X, predictions, color='blue', linewidth=2, label='Linear Regression Line')
plt.title('Linear Regression', fontsize=18, fontweight='bold')
plt.xlabel('X', fontsize=14)
plt.ylabel('Predicted Y', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.show()
