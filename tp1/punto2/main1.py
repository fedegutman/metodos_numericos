from scipy.interpolate import lagrange, CubicSpline
import matplotlib.pyplot as plt
import numpy as np

def get_coordinates(file:str) -> list:
    data = []
    with open(file, 'r') as f:
        for line in f:
            x, y = line.strip().split(" ")
            data.append((x,y))
    return data

# Calculo la trayectoria real
groundtruth = get_coordinates("tp1/punto2/mnyo_ground_truth.csv")
x1 = np.array([float(x) for x, _ in groundtruth])
y1 = np.array([float(y) for _, y in groundtruth])

# Estimo la trayectoria interpolando de manera lineal
estimated = get_coordinates("tp1/punto2/mnyo_mediciones.csv")
x2 = np.array([float(x) for x, _ in estimated])
y2 = np.array([float(y) for _, y in estimated])

# Estimo la trayectoria interpolando con lagrange
tiempo1 = np.linspace(0, max(x2), 10)
tiempo2 = np.linspace(0, max(x2), 500)

poly1 = lagrange(tiempo1, x2)
poly2 = lagrange(tiempo1, y2)

# Estimo la trayectoria interpolando con splines cúbicos
spline1 = CubicSpline(tiempo1, x2)
spline2 = CubicSpline(tiempo1, y2)

plt.plot(x1, y1, label='Ground truth', color="black")
plt.plot(x2, y2, label='Linear', color='red', linestyle='dashed')
plt.plot(poly1(tiempo2), poly2(tiempo2), label='Lagrange', color='blue', linestyle='dotted')
plt.plot(spline1(tiempo2), spline2(tiempo2), label='Cubic Spline', color='green', linestyle='dashdot')
plt.legend()
plt.title('Ground truth vs Interpolated trajectories', fontsize=20)
plt.show()