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


dNdt = lambda N, P, t: r*N*(1 - (N/K)) - alpha*N*P
dPdt = lambda N, P, t: beta*N*P - q*P

alpha = 0.3
beta = 0.5
r = 0.8
q = 0.3
K = 20

# Grafico las trayectorias del sistema
figure, axis = plt.subplots(2, 1)

N0, P0 = 5, 5
t0 = 0
tf = np.linspace(1, 100, 50)
h = 0.01

prey, predator = [], []

for t in tf:
    n = int((t-t0)/h)
    prey_new, predator_new = runge_kutta4_system(dNdt, dPdt, t0, N0, P0, h, n)
    prey.append(prey_new)
    predator.append(predator_new)

axis[0].plot(tf, prey, label='Presa', color='blue')
axis[0].set_xlabel('Tiempo')
axis[0].set_ylabel('Población de presas')

axis[1].plot(tf, predator, label='Predador', color='red')
axis[1].set_xlabel('Tiempo')
axis[1].set_ylabel('Población de predadores')

figure.subplots_adjust(hspace=0.3)
plt.show()