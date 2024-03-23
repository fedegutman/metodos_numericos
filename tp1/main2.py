from scipy.interpolate import lagrange
import matplotlib.pyplot as plt
import numpy as np

def get_coordinates(file:str) -> list:
    data = []
    with open(file, 'r') as f:
        for line in f:
            x, y = line.strip().split(" ")
            data.append((x,y))
    return data


coordinates = get_coordinates("tp1/mnyo_mediciones.csv")
x = [float(x) for x, _ in coordinates]
y = [float(y) for _, y in coordinates]

poly = lagrange(x, y)

x_values = np.linspace(min(x), max(x), 500)
y_values = poly(x_values)

plt.plot(x_values, y_values, label='Interpolated polynomial')
plt.scatter(x, y, color='red', label='Original data')
plt.legend()
plt.show()

# Docs scipy -----> dhttps://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.lagrange.html