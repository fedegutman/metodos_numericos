import numpy as np
from matplotlib.pylab import plt
from scipy.interpolate import lagrange

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

# Comparo lagrange con nodos de chebyshev a lagrange sin nodos
chebyshev_lagrange = lagrange(nodes, f(nodes))

xi = np.array(np.linspace(-4, 4, 17))
lagrange_poly = lagrange(xi, f(xi))

plt.plot(xf, yf, label='Original function', color="black")
plt.plot(xf, chebyshev_lagrange(xf), label='Lagrange chebyshev', color='green', linestyle='dashed')
plt.plot(xf, lagrange_poly(xf), label='Lagrange equispaced', color='blue', linestyle='dotted')
plt.scatter(nodes, f(nodes), color='black')
plt.ylim(0,4)
plt.legend()
plt.title("Lagrange interpolation using Chebyshev Nodes vs Equispaced Nodes", fontsize=20, y=1.03)
plt.show()

# falta graficar las demas cosas