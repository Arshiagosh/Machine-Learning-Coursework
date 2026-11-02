import numpy as np
import matplotlib.pyplot as plt

x = np.array([0, 10, 20, 30, 40, 50, 60])  
y = np.array([16, 27, 28, 39, 39, 48, 51])

plt.scatter(x, y, color='blue', label="Data points")

plt.xlabel("Minutes after 7:00 AM (x)")
plt.ylabel("Travel Time (y)")
plt.title("Scatter Plot")
plt.legend()
plt.grid(True)
plt.show()

