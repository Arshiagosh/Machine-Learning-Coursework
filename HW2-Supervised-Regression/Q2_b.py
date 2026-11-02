import numpy as np
import matplotlib.pyplot as plt

x = np.array([0, 10, 20, 30, 40, 50, 60])
y = np.array([16, 27, 28, 39, 39, 48, 51])

X = np.vstack((np.ones(len(x)), x)).T

# theta = (X^T * X)^(-1) * X^T * Y
theta = np.linalg.inv(X.T @ X) @ X.T @ y

# (a) and (b)
intercept = theta[0]
slope = theta[1]

regression_eq_matrix = f"y = {intercept:.2f} + {slope:.2f}x"
print("Regression Equation:", regression_eq_matrix)

plt.scatter(x, y, color='blue', label="Data points")
plt.plot(x, intercept + slope*x, color='red', label="Regression Line")

plt.xlabel("Minutes after 7:00 AM (x)")
plt.ylabel("Travel Time (y)")
plt.title("Scatter Plot")
plt.legend()
plt.grid(True)
plt.show()