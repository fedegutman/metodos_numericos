import numpy as np
import matplotlib.pyplot as plt

def runge_kutta4_system(f1, f2, t0, x0, y0, h, n):
    t = t0
    x = x0
    y = y0
    for i in range(n):
        k1 = h*f1(x, y, t)
        l1 = h*f2(x, y, t)
        
        k2 = h*f1(x + k1/2, y + l1/2, t + h/2)
        l2 = h*f2(x + k1/2, y + l1/2, t + h/2)
        
        k3 = h*f1(x + k2/2, y + l2/2, t + h/2)
        l3 = h*f2(x + k2/2, y + l2/2, t + h/2)
        
        k4 = h*f1(x + k3, y + l3, t + h)
        l4 = h*f2(x + k3, y + l3, t + h)
        
        x = x + (k1 + 2*k2 + 2*k3 + k4)/6
        y = y + (l1 + 2*l2 + 2*l3 + l4)/6
        t = t + h
    return x, y

# Define las ecuaciones diferenciales del sistema de Lotka-Volterra
dNdt = lambda N, P, t: r*N - alpha*N*P

# Predador
dPdt = lambda N, P, t: beta*N*P - q*P


beta  = 0.3
q = 1.1

# Parámetros del modelo
r = 0.5
K = 100
m = 0.2

# Condiciones iniciales
N0 = 10
P0 = 10

# Tiempo inicial, final y paso
t0 = 0
tf = np.linspace(0, 10, 1000)
h = 0.01

# Valores de alpha para iterar
alphas = np.linspace(0.1, 1, 10)

# Graficar el diagrama de fases para cada valor de alpha
plt.figure(figsize=(8, 6))
plt.xlabel('Población de presas')
plt.ylabel('Población de predadores')
plt.title('Diagrama de Fases - Modelo de Lotka-Volterra')

for alpha in alphas:
    prey, predator = [], []
    for t in tf:
        n = int((t - t0) / h)
        prey.append(runge_kutta4_system(dNdt, dPdt, t0, N0, P0, h, n)[0])
        predator.append(runge_kutta4_system(dNdt, dPdt, t0, N0, P0, h, n)[1])
    plt.plot(prey, predator, label=f'alpha = {alpha}')

plt.legend()
plt.grid(True)
plt.show()
