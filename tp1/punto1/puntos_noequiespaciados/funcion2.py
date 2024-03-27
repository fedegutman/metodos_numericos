from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt

def chebyshev_nodes(n, interval):
    '''
    n: cantidad de nodos
    interval: tupla con los extremos del intervalo
    '''
    a, b = interval
    k = np.arange(1, n + 1)
    nodes = 0.5 * ((b - a) * np.cos((2 * k - 1) * np.pi / (2 * n)) + (a + b))
    return nodes

def f(x1, x2):
    '''
    x1, x2 ∈ [-1,1]
    '''
    return 0.75*np.exp( (-((10*x1-2)**2)/4) - (((9*x2-2)**2)/4) ) + 0.65*np.exp( (-((9*x1+1)**2)/9) - (((10*x2+1)**2)/2) ) + 0.55*np.exp( (-((9*x1-6)**2)/4) - (((9*x2-3)**2)/4) ) - 0.01*np.exp( (-((9*x1-7)**2)/4) - (((9*x2-3)**2)/4) )

xf = np.linspace(-1, 1, 500)
yf = np.linspace(-1, 1, 500)
X1, X2 = np.meshgrid(xf, yf)
Z = f(X1, X2)

nodes_x = chebyshev_nodes(20, (-1,1))
nodes_x = np.append(nodes_x, [-1, 1])
nodes_x = np.sort(nodes_x)

nodes_y = chebyshev_nodes(20, (-1,1))
nodes_y = np.append(nodes_y, [-1, 1])
nodes_y = np.sort(nodes_y)

XI, YI = np.meshgrid(nodes_x, nodes_y)

columna = np.column_stack((XI.flatten(), YI.flatten()))

# Grafico la funcion original

fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(221, projection='3d')
ax1.plot_surface(X1, X2, Z, cmap='viridis')
ax1.set_title('Original Function')

# Interpolo la funcion f usando grid data (cubic)
Zi_cubic = griddata(columna, f(XI, YI).flatten(), (X1, X2), method='cubic')
ax2 = fig.add_subplot(222, projection='3d')  # 122 means 1 row, 2 columns, second plot
ax2.plot_surface(X1, X2, Zi_cubic, cmap='viridis')
ax2.set_title('Cubic Interpolation')

fig.suptitle("3D Interpolation Function B using Chebyshev Nodes", fontsize=20, y=0.97)

# Interpolo la funcion f usando grid data (nearest)
Zi_nearest = griddata(columna, f(XI, YI).flatten(), (X1, X2), method='nearest')
ax3 = fig.add_subplot(223, projection='3d')
ax3.plot_surface(X1, X2, Zi_nearest, cmap='viridis')
ax3.set_title('Nearest Interpolation')

# Interpolo la funcion f usando grid data (linear)
Zi_linear = griddata(columna, f(XI, YI).flatten(), (X1, X2), method='linear')
ax4 = fig.add_subplot(224, projection='3d') # 224 is 2 rows, 2 columns, fourth plot
ax4.plot_surface(X1, X2, Zi_linear, cmap='viridis')
ax4.set_title('Linear Interpolation')

plt.show()

# Grafico el error de cada interpolacion
fig2 = plt.figure(figsize=(12, 6))

# Error de la interpolacion cubica
Zi_cubic_error = abs(Zi_cubic - Z)
ax1 = fig2.add_subplot(131, projection='3d')
ax1.plot_surface(X1, X2, Zi_cubic_error, cmap='viridis')
ax1.set_title('Error of Cubic Interpolation')

# Error de la interpolacion nearest
Zi_nearest_error = abs(Zi_nearest - Z)
ax2 = fig2.add_subplot(132, projection='3d')
ax2.plot_surface(X1, X2, Zi_nearest_error, cmap='viridis')
ax2.set_title('Error of Nearest Interpolation')

# Error de la interpolacion lineal
Zi_linear_error = abs(Zi_linear - Z)
ax3 = fig2.add_subplot(133, projection='3d')
ax3.plot_surface(X1, X2, Zi_linear_error, cmap='viridis')
ax3.set_title('Error of Linear Interpolation')

fig2.suptitle("3D Interpolation Error Function B using Chebyshev nodes", fontsize=20, y=0.93)

plt.show()