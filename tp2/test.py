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

# Defino las ecuaciones de competencia de Lotka-Volterra

# Especie 1
dN1dt = lambda N1, N2, t: r1*N1*(1 - (N1 + alpha*N2)/K1)
'alpha -> el efecto que tiene un individuo especie 2 sobre el crecimiento poblacional de la especie 1'
'por eso es que lo multiplico por N2 (la cantidad de individuos de la especie 2)'

# Especie 2
dN2dt = lambda N1, N2, t: r2*N2*(1 - (N2 + beta*N1)/K2)
'beta -> el efecto que tiene la especie 1 sobre el crecimiento poblacional de la especie 2'

# Caso 2 -> Gana la especie 2
alpha = 1.5
beta = 0.5
K1, K2 = 70, 60
r1, r2 = 0.3, 0.3
N1_0, N2_0 = 10, 10
t0, h = 0, 0.1

tf = np.linspace(1, 100, 100)

figure, axis = plt.subplots()

N1 = []
N2 = []

for t in tf:
    n = int((t-t0)/h)
    N1.append(runge_kutta4_system(dN1dt, dN2dt, t0, N1_0, N2_0, h, n)[0])
    N2.append(runge_kutta4_system(dN1dt, dN2dt, t0, N1_0, N2_0, h, n)[1])
    N1_0 = N1[-1]
    N2_0 = N2[-1]
    t0 = t

axis.plot(tf, np.array(N1), label='Especie 1', color='blue')
axis.plot(tf, np.array(N2), label='Especie 2', color='green')
axis.set_title('Gana especie 2')

plt.show()
