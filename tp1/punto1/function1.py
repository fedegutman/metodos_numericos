import numpy as np
import scipy.interpolate as spi
import matplotlib.pyplot as plt

def f(x):
    '''
    x ∈ [-4,4]
    '''
    return (0.3**abs(x)) * np.sin(4*x) - np.tanh(2*x) + 2

xf = np.array(np.linspace(-4, 4, 500))
yf = np.array([f(i) for i in xf])

' -------- Primero tomando puntos equiespaciados -------- '

xi = np.array(np.linspace(-4, 4, 15))
yi = np.array([f(i) for i in xi])

figure, axis = plt.subplots(2, 2)
axis[0,0].plot(xf, yf)
axis[0,0].set_title("Original function")

# Utilizo interpolacion lineal para interpolar los puntos

f_linear = spi.interp1d(xi, yi, kind='linear')

axis[0,1].plot(xf, f_linear(xf), color='orange')
axis[0,1].set_title("Linear interpolation")

# 'Utilizo polinomios de lagrange para interpolar los puntos'
lagrange_poly = spi.lagrange(xi, yi)

axis[1,0].plot(xf, lagrange_poly(xf), color='green')
axis[1,0].set_title("Lagrange polynomial interpolation")
axis[1,0].set_ylim(0,4)

# Utilizo splines cubicos para interpolar los puntos
cubic_spline = spi.CubicSpline(xi, yi)

axis[1,1].plot(xf, cubic_spline(xf), color='red')
axis[1,1].set_title("Cubic spline interpolation")


# Grafico los nodos
for row in axis:
    for subplot in row:
        subplot.scatter(xi, yi, color='black')

figure.suptitle("Interpolation Methods Function A", fontsize=20)
plt.show()

# Grafico todos los metodos juntos
plt.plot(xf, yf, label='Original function', color="black")
plt.plot(xf, f_linear(xf), label='Linear interpolation', color='orange', linestyle='dotted')
plt.plot(xf, lagrange_poly(xf), label='Lagrange polynomial interpolation', color='green', linestyle='dashed')
plt.plot(xf, cubic_spline(xf), label='Cubic spline interpolation', color='red', linestyle="dashdot")
plt.scatter(xi, yi, color='black')
plt.ylim(0,4)

plt.legend()
plt.title("Interpolation Methods Overlapped Function A", fontsize=20)
plt.show()

# cambiar lineal por chebyshev y calcular error
# cuantos puntos usar para interpolar
# como hacer con los puntos no equiespaciados