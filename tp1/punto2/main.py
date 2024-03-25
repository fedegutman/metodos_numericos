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

# Primero grafico la trayectoria real

groundtruth = get_coordinates("tp1/punto2/mnyo_ground_truth.csv")
x1 = np.array([float(x) for x, _ in groundtruth])
y1 = np.array([float(y) for _, y in groundtruth])

figure, axis = plt.subplots(1,2)
axis[0].plot(x1, y1)
axis[0].set_title("Ground Truth")

# Luego grafico la trayectoria estimada interpolando
estimated = get_coordinates("tp1/punto2/mnyo_mediciones.csv")
x2 = np.array([float(x) for x, _ in estimated])
y2 = np.array([float(y) for _, y in estimated])

poly = lagrange(x2, y2)

x_values = np.linspace(min(x2), max(x2), 500)
y_values = poly(x_values)

axis[1].plot(x_values, y_values)
axis[1].set_title("Lagrange polynomial interpolation")
axis[1].scatter(x2, y2, color='red')
axis[1].set_ylim(0,6)
plt.legend()
plt.show()

# Finalmente grafico la trayectoria real y la estimada en el mismo grafico
plt.plot(x1, y1, label='Ground truth', color="black")
plt.plot(x_values, y_values, label='Interpolated polynomial', color='blue', linestyle='dotted')
plt.title('Ground truth vs Interpolated polynomial', fontsize=20)
plt.scatter(x2, y2, color='red')
plt.ylim(0,6)
plt.legend()
plt.show()