import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Sample Data
X = np.array([[1.0], [1.5], [5.0], [8.0], [1.0], [9.0]])
y = np.array([0, 0, 1, 1, 0, 1])

# Polynomial Features
degree = 3
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X)

# Fit Polynomial Regression Model
model = LinearRegression()
model.fit(X_poly, y)

# Generate Predictions
X_range = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
X_range_poly = poly.transform(X_range)
y_pred = model.predict(X_range_poly)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Data Points', edgecolor='k')
plt.plot(X_range, y_pred, color='red', linewidth=2, label=f'Polynomial Regression (degree={degree})')

# Beautify the plot
plt.title('Polynomial Regression', fontsize=16, fontweight='bold')
plt.xlabel('Feature', fontsize=14)
plt.ylabel('Target', fontsize=14)
plt.legend()
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.gca().set_facecolor('white')
plt.tight_layout()

# Show Plot
plt.show()
