import matplotlib.pyplot as plt
import numpy as np
from math import exp

# MÉTODOS

def runge_kutta4(f, t0, y0, h, n):
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

def midpoint_runge_kutta(f, t0, y0, h, n):
    t = t0
    y = y0
    for i in range(n):
        k1 = h*f(t, y)
        k2 = h*f(t + h/2, y + k1/2)
        y = y + k2
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
dNdt_exp = lambda t, N: r*N

# Crecimiento logistico
dNdt_log = lambda t, N: r*N * ((K - N)/K)

# Obtengo las soluciones analíticas
Nexp = lambda N0, r, t: N0 * exp(r*t)
Nlog = lambda N0, r, K, t: K / (1 + (K/N0 - 1) * exp(-r*t))

# Grafico tamaño poblacional en función del tiempo (N vs t) y estudió la variación de los parámetros
t = np.linspace(0, 100, 1000)
figure, axis = plt.subplots(1, 3)

# Parámetros 1
N0 = 2
r = 0.01
K = 15

poblacion_exp = np.array([Nexp(N0, r, i) for i in t])
poblacion_log = np.array([Nlog(N0, r, K, i) for i in t])

axis[0].plot(t, poblacion_exp, label=f'Crecimiento exponencial')
axis[0].plot(t, poblacion_log, label=f'Crecimiento logístico')
axis[0].legend()
axis[0].set_xlabel('Tiempo')
axis[0].set_ylabel('Población')
axis[0].set_title(f'N0={N0}, r={r}, K={K}')

# Parámetros 2
N0 = 1
r = 0.05
K = 20

poblacion_exp = np.array([Nexp(N0, r, i) for i in t])
poblacion_log = np.array([Nlog(N0, r, K, i) for i in t])

axis[1].plot(t, poblacion_exp, label=f'Crecimiento exponencial')
axis[1].plot(t, poblacion_log, label=f'Crecimiento logístico')
axis[1].set_title(f'N0={N0}, r={r}, K={K}')

# Parámetros 3

N0 = 20
r = -0.1
K = 50

poblacion_exp = np.array([Nexp(N0, r, i) for i in t])
poblacion_log = np.array([Nlog(N0, r, K, i) for i in t])

axis[2].plot(t, poblacion_exp, label=f'Crecimiento exponencial')
axis[2].plot(t, poblacion_log, label=f'Crecimiento logístico')
axis[2].set_title(f'N0={N0}, r={r}, K={K}')

plt.show()

# Grafico la variación poblacional en función del tamaño poblacional (dN/dt vs N) y estudió la variación de los parámetros
N = np.array(range(1, 51))

figure, axis = plt.subplots(1, 3)

# Parámetros 1
r = 0.5
K = 20

variacion_exp = np.array([r * i for i in N])
variacion_log = np.array([r * i * ((K - i)/K) for i in N])

axis[0].plot(N, variacion_exp, label='Crecimiento exponencial')
axis[0].plot(N, variacion_log, label='Crecimiento logístico')
axis[0].legend()
axis[0].set_xlabel('Población')
axis[0].set_ylabel('Variación poblacional')
axis[0].set_title(f'r={r}, K={K}')

# Parámetros 2
r = 0.1
K = 70

variacion_exp = np.array([r * i for i in N])
variacion_log = np.array([r * i * ((K - i)/K) for i in N])

axis[1].plot(N, variacion_exp, label='Crecimiento exponencial')
axis[1].plot(N, variacion_log, label='Crecimiento logístico')
axis[1].set_title(f'r={r}, K={K}')

# Parámetros 3
r = -0.1
K = 50

variacion_exp = np.array([r * i for i in N])
variacion_log = np.array([r * i * ((K - i)/K) for i in N])

axis[2].plot(N, variacion_exp, label='Crecimiento exponencial')
axis[2].plot(N, variacion_log, label='Crecimiento logístico')
axis[2].set_title(f'r={r}, K={K}')
plt.show()

# PUNTO DE EQUILIBRIO -> CUANDO R = 0 o CUANDO N = K (PREGUNTAR)

# Obtengo las soluciones numéricas de ambas ecuaciones por los métodos vistos y comparo con las soluciones exactas
N0 = 2
r = 0.1
K = 15
t = 1
h = 0.1
n = int(t / h)

exact_exp = Nexp(N0, r, t)
runge_kutta_exp = runge_kutta4(dNdt_exp, 0, N0, h, n)
midpoint_runge_kutta_exp = midpoint_runge_kutta(dNdt_exp, 0, N0, h, n)
euler_exp = euler(dNdt_exp, 0, N0, h, n)
print('MODELO EXPONENCIAL')
print(f'Exacta: {exact_exp}, runge kutta: {runge_kutta_exp}, euler: {euler_exp}, runge kutta de punto medio: {midpoint_runge_kutta_exp}')
print(f'Diferencia entre exacta y runge kutta: {abs(exact_exp - runge_kutta_exp)}')
print(f'Diferencia entre exacta y runge kutta de punto medio: {abs(exact_exp - midpoint_runge_kutta_exp)}')
print(f'Diferencia entre exacta y euler: {abs(exact_exp - euler_exp)}\n\n')

exact_log = Nlog(N0, r, K, t)
runge_kutta_log = runge_kutta4(dNdt_log, 0, N0, h, n)
midpoint_runge_kutta_log = midpoint_runge_kutta(dNdt_log, 0, N0, h, n)
euler_log = euler(dNdt_log, 0, N0, h, n)

print('MODELO LOGISTICO')
print(f'Exacta: {exact_log}, runge kutta: {runge_kutta_log}, euler: {euler_log}, runge kutta de punto medio: {midpoint_runge_kutta_log}')
print(f'Diferencia entre exacta y runge kutta: {abs(exact_log - runge_kutta_log)}')
print(f'Diferencia entre exacta y runge kutta de punto medio: {abs(exact_log - midpoint_runge_kutta_log)}')
print(f'Diferencia entre exacta y euler: {abs(exact_log - euler_log)}')

# Busco punto de equilibrio/punto fijo de la ecuacion logistica
'''
dNdt_log = 0 ----> 0 = r*N * ((K - N)/K)
Entonces r*N = 0 o K - N = 0
Recuerdo, r = tasa de crecimiento, n = tamaño de la población, K = tamaño máximo de la población (NO PUEDE SER CERO)
Entonces r = 0 (tasa de crecimiento nula) N = 0 (arranco sin poblacion) o N = K (se alcanza el tamaño máximo de la población)
'''

# Corroboro
# r = 0
r = 0
N = 2
K = 15
print(f'Para r = 0 (N = 0, K = 15): {r*N * ((K - N)/K)}')

# N = 0
r = 0.1
N = 0
K = 15
print(f'Para N = 0 (r = 0.1, K = 15): {r*N * ((K - N)/K)}')

# N = K
r = 0.1
N = 15
K = 15
print(f'Para N = K (r = 0.1, K = 15): {r*N * ((K - N)/K)}')
