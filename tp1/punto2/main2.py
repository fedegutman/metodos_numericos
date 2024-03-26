import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange

def get_coordinates(file:str) -> list:
    data = []
    with open(file, 'r') as f:
        for line in f:
            x, y = line.strip().split(" ")
            data.append((x,y))
    return data

# Grafico la trayectoria del primer vehiculo
first_vehicle = get_coordinates("tp1/punto2/mnyo_ground_truth.csv")
x1 = np.array([float(x) for x, _ in first_vehicle])
y1 = np.array([float(y) for _, y in first_vehicle])

plt.plot(x1, y1, color='black', label='First Vehicle')

# Interpolo la trayectoria del segundo vehiculo usano polinomios de lagrange y grafico
second_vehicle = get_coordinates("tp1/punto2/mnyo_mediciones2.csv")
x2 = np.array([float(x) for x, _ in second_vehicle])
y2 = np.array([float(y) for _, y in second_vehicle])

lagrange_poly = lagrange(x2, y2)

plt.plot(x1, lagrange_poly(x1), color='blue', label='Second Vehicle')
plt.title('Trajectories of both vehicles', fontsize=20)
plt.legend()
plt.show()

# Busco la interseccion entre las trayectorias
