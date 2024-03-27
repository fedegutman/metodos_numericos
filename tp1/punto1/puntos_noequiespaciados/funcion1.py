import numpy as np
from matplotlib.pylab import plt
from scipy.interpolate import lagrange, interp1d, CubicSpline

# Utilizo nodos de chebyshev para reducir las oscilaciones en los extremos

def chebyshev_nodes(n, interval):
    '''
    n: cantidad de nodos
    interval: tupla con los extremos del intervalo
    '''
    a, b = interval
    k = np.arange(1, n + 1)
    nodes = 0.5 * ((b - a) * np.cos((2 * k - 1) * np.pi / (2 * n)) + (a + b))
    return nodes

def f(x):
    '''
    x ∈ [-4,4]
    '''
    return (0.3**abs(x)) * np.sin(4*x) - np.tanh(2*x) + 2

xf = np.array(np.linspace(-4, 4, 500))
yf = f(xf)

nodes = chebyshev_nodes(20, (-4,4))
nodes = np.append(nodes, [-4, 4])
nodes = np.sort(nodes)
nodes.sort()

# Comparo lagrange con nodos de chebyshev a lagrange sin nodos
chebyshev_lagrange = lagrange(nodes, f(nodes))

xi = np.array(np.linspace(-4, 4, 20))
lagrange_poly = lagrange(xi, f(xi))

plt.plot(xf, yf, label='Original function', color="black")
plt.plot(xf, chebyshev_lagrange(xf), label='Lagrange chebyshev', color='green', linestyle='dashed')
plt.plot(xf, lagrange_poly(xf), label='Lagrange equispaced', color='blue', linestyle='dotted')
plt.scatter(nodes, f(nodes), color='black')
plt.ylim(0,4)
plt.legend()
plt.title("Lagrange interpolation using Chebyshev Nodes vs Equispaced Nodes", fontsize=20, y=1.03)
plt.show()

# Grafico los tres metodos de interpolacion usados anteriormente pero ahora con nodos de chebyshev
f_linear = interp1d(nodes, f(nodes), kind='linear')
cubic_spline = CubicSpline(nodes, f(nodes))

plt.plot(xf, yf, label='Original function', color="black")
plt.plot(xf, f_linear(xf), label='Linear interpolation', color='orange', linestyle='dotted')
plt.plot(xf, chebyshev_lagrange(xf), label='Lagrange interpolation', color='green', linestyle='dashed')
plt.plot(xf, cubic_spline(xf), label='Cubic spline interpolation', color='red', linestyle="dashdot")
plt.scatter(nodes, f(nodes), color='black')
plt.ylim(0,4)
plt.legend()
plt.title("Overlapped Interpolation Methods Using Chebyshev nodes Function A", fontsize=20)
plt.show()

# Grafico los errores en los metodos de interpolacion
figure, axis = plt.subplots(1, 3)
axis[0].plot(xf, abs(yf - f_linear(xf)), color='orange')
axis[0].set_title("Linear interpolation error")
axis[0].set_ylim(0, 0.3)

axis[1].plot(xf, abs(yf - chebyshev_lagrange(xf)), color='green')
axis[1].set_title("Lagrange interpolation error")
axis[1].set_ylim(0, 0.3)

axis[2].plot(xf, abs(yf - cubic_spline(xf)), color='red')
axis[2].set_title("Cubic spline interpolation error")
axis[2].set_ylim(0, 0.3)

figure.suptitle("Interpolation Methods Errors Using Chebyshev nodes Function A", fontsize=20)
plt.show()