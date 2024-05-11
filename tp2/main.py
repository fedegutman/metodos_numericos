import matplotlib.pyplot as plt
import numpy as np
from math import exp

# MÉTODOS

def runge_kutta(f, t0, y0, h, n):
    t = t0
    y = y0
    for i in range(n):
        k1 = h*f(t, y)
        k2 = h*f(t + h/2, y + k1/2)
        k3 = h*f(t + h/2, y + k2/2)
        k4 = h*f(t + h, y + k3)
        y = y + (k1 + 2*k2 + 2*k3 + k4)/6
        t = t + h
    return y

def euler(f, t0, y0, h, n):
    t = t0
    y = y0
    for _ in range(n):
        y += h * f(t, y)
        t += h
    return y

# ODES -> N(t) es el tamaño de la población en el tiempo t

# Crecimiento exponencial
dNdt_exp = lambda r, N: r*N

'''
-> El crecimiento de la población está determinado por una tasa instantánea de crecimiento per cápita r

-> N0 la condición incial, es decir el tamaño de la población en el tiempo t = 0
'''

# Crecimiento logistico
dNdt_log = lambda r, N, K: r*N * ((K - N)/K)

'''
-> K es la capacidad de carga del sistema, es decir el tamaño máximo de la población que el sistema puede soportar
-> Además se pide que la tasa de crecimiento per cápita, (1/N) (dN/dt), dependa del tamaño de la población.
'''

# Obtengo las soluciones analíticas
Nexp = lambda N0, r, t: N0 * exp(r*t)
Nlog = lambda N0, r, K, t: K / (1 + (K/N0 - 1) * exp(-r*t))

N0 = 2 
r = 0.01 # o 0.1 probar
K = 100 # probar con otros valores

# Grafico tamaño poblacional en función del tiempo (N vs t) 
t = np.linspace(0, 100, 1000)
poblacion_exp = np.array([Nexp(N0, r, i) for i in t])
poblacion_log = np.array([Nlog(N0, r, K, i) for i in t])
'chequear parametros'

plt.plot(t, poblacion_exp, label='Crecimiento exponencial')
plt.plot(t, poblacion_log, label='Crecimiento logístico')
plt.legend()
plt.xlabel('Tiempo')
plt.ylabel('Población')
plt.title('Soluciones analíticas')
plt.show()

# Grafico la variación poblacional en función del tamaño poblacional (dN/dt vs N)

# ESTO ESTA HECHO CON CHAT (CHEQUEAR)

N = np.linspace(0, 100, 1000)
variacion_exp = 0.1 * N
variacion_log = 0.1 * N * ((100 - N)/100)

plt.plot(N, variacion_exp, label='Crecimiento exponencial')
plt.plot(N, variacion_log, label='Crecimiento logístico')
plt.legend()
plt.xlabel('Población')
plt.ylabel('Variación poblacional')
plt.title('Soluciones analíticas')
plt.show()

# Obtengo las soluciones numéricas de ambas ecuaciones por los métodos vistos y comparo con las soluciones exactas
