import matplotlib.pyplot as plt
import numpy as np

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

dNdt = lambda N, P, t: r*N - alpha*N*P
dPdt = lambda N, P, t: beta*N*P - q*P

alpha = 0.9
beta = 0.8
q = 1.2
r = 1.1
N0 = 0.9
P0 = 0.8

t0 = 0
tf = 10
h = 0.01
n = int((tf - t0) / h)

prey, predator = runge_kutta4_system(dNdt, dPdt, t0, N0, P0, h, n)

plt.figure()
plt.plot(prey, predator)
plt.xlabel('Población de presas')
plt.ylabel('Población de predadores')
plt.title('Diagrama de fases del modelo de Lotka-Volterra')
plt.show()