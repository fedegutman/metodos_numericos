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

xi = np.array(np.linspace(-4, 4, 10))
yi = np.array([f(i) for i in xi])

figure, axis = plt.subplots(2, 3)
axis[0,0].plot(xf, yf)
axis[0,0].set_title("Original function")

# Utilizo interpolacion lineal para interpolar los puntos
f_linear = spi.interp1d(xi, yi, kind='linear')

axis[0,1].plot(xf, f_linear(xf), color='orange')
axis[0,1].set_title("Linear interpolation")

# 'Utilizo polinomios de lagrange para interpolar los puntos'
lagrange_poly = spi.lagrange(xi, yi)

axis[0,2].plot(xf, lagrange_poly(xf), color='green')
axis[0,2].set_title("Lagrange polynomial interpolation")

# Utilizo splines cubicos para interpolar los puntos
cubic_spline = spi.CubicSpline(xi, yi)

axis[1,0].plot(xf, cubic_spline(xf), color='red')
axis[1,0].set_title("Cubic spline interpolation")

# Muestro la funcion original y el polinomio de lagrange
axis[1,1].plot(xf, yf)
axis[1,1].plot(xf, lagrange_poly(xf), label='Lagrange polynomial interpolation', color='green')
axis[1,1].set_title("Lagrange over original")

# Muestro la funcion original y la interpolacion de splines cubicos
axis[1,2].plot(xf, yf)
axis[1,2].plot(xf, cubic_spline(xf), color='red')
axis[1,2].set_title("Splines over original")

figure.suptitle("Interpolation Methods Function A", fontsize=20)
plt.show()

# hacer 00 normal, 01 lagrange, 10 chebychev, 11 splines y otro con todos solapados