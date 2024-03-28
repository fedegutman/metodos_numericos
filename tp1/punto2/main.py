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
tiempo1 = np.linspace(0, max(x2), len(x2))
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

# Grafico la trayectoria del primer vehiculo y sus interpolaciones
plt.plot(x1, y1, label='Ground truth', color="black")
plt.plot(poly1(tiempo2), poly2(tiempo2), label='Lagrange', color='red')
plt.plot(spline1(tiempo2), spline2(tiempo2), label='Cubic Spline', color='green')

# Interpolo la trayectoria del segundo vehiculo usando splines cúbicos y grafico
second_vehicle = get_coordinates("tp1/punto2/mnyo_mediciones2.csv")
x3 = np.array([float(x) for x, _ in second_vehicle])
y3 = np.array([float(y) for _, y in second_vehicle])

tiempo1 = np.linspace(0, max(x2), len(x3))

spline3_x = CubicSpline(tiempo1, x3)
spline3_y = CubicSpline(tiempo1, y3)

plt.plot(spline3_x(tiempo2), spline3_y(tiempo2), label='Second Vehicle', color='blue')

# Busco la interseccion entre las trayectorias utilizando el metodo de newton raphson

def f(t1, t2):
    return np.array([spline1(t1) - spline3_x(t2), spline2(t1) - spline3_y(t2)])

def jacobian(t1, t2):
    return np.array([[spline1.derivative()(t1), -spline3_x.derivative()(t2)], [spline2.derivative()(t1), -spline3_y.derivative()(t2)]])

def newton_raphson(estimation, function, jac, tolerance = 1e-20, max_iteration = 100):
    for i in range(max_iteration):
        delta = np.linalg.solve(jac(estimation[0], estimation[1]), function(estimation[0], estimation[1]))
        estimation -= delta
        if np.linalg.norm(delta) < tolerance:
            return estimation
    return estimation

estimation = np.array([0.0, 0.0])
intersection = newton_raphson(estimation, f, jacobian)

plt.plot(spline1(intersection[0]), spline2(intersection[0]), 'yo', label='Intersection')

plt.title('Trajectories of both vehicles', fontsize=20)
plt.legend()
plt.show()

