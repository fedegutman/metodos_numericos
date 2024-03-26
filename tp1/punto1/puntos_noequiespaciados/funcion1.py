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
yf = np.array([f(i) for i in xf])

figure, axis = plt.subplots(1,2)

axis[0].plot(xf, yf)
axis[0].set_title("Original function")

# Utilizo polinomios de lagrange para interpolar los nodos de chebyshev

nodes = chebyshev_nodes(20, (-4,4))

chebyshev_lagrange = lagrange(nodes, f(nodes))

axis[1].plot(xf, chebyshev_lagrange(xf), color='green')
axis[1].set_title("Lagrange polynomial interpolation")
axis[1].set_ylim(0,4)

figure.suptitle("Lagrange interpolation using Chebyshev Nodes", fontsize=20)
plt.show()

# Comparo lagrange con nodos de chebyshev a lagrange sin nodos

xi = np.array(np.linspace(-4, 4, 17))
lagrange_poly = lagrange(xi, f(xi))

plt.plot(xf, yf, label='Original function', color="black")
plt.plot(xf, chebyshev_lagrange(xf), label='Lagrange chebyshev', color='green', linestyle='dashed')
plt.plot(xf, lagrange_poly(xf), label='Lagrange equispaced', color='blue', linestyle='dotted')
plt.scatter(nodes, f(nodes), color='black')
plt.ylim(0,4)
plt.legend()
plt.title("Lagrange interpolation using Chebyshev Nodes vs Equispaced Nodes", fontsize=20)
plt.show()

# falta graficar las demas cosas