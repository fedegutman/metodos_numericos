import matplotlib.pyplot as plt
import numpy as np

def runge_kutta4_system_vectors(f1, f2, t0, x0, y0, h, n):
    t_values = [t0]
    x_values = [x0]
    y_values = [y0]
    
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

        x_values.append(x)
        y_values.append(y)
        t_values.append(t)
        
    return t_values, x_values, y_values

def runge_kutta4_system(f1, f2, t0, x0, y0, h, n):
    t = t0
    x = x0
    y = y0
    for i in range(n):
        k1 = h*f1(x, y, t)
        l1 = h*f2(x, y, t)
        
        k2 = h*f1(x + k1/2, y + l1/2, t + h/2)
        l2 = h*f2(t + k1/2, y + l1/2, t + h/2)
        
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

# Parámetros
r1, r2 = 0.5, 0.7
alpha, beta = 0.5, 0.8
K1, K2 = 100, 130

# Condiciones iniciales
N1_0 = 10
N2_0 = 10

# Tiempo inicial
t0 = 0

# Paso
h = 0.01
n = 1000

final_N1, final_N2 = runge_kutta4_system(dN1dt, dN2dt, t0, N1_0, N2_0, h, n)
print(f'N1: {final_N1}, N2: {final_N2}')

# -> Cuantos parametros usar?

# Grafico las isoclinas 
'''
0 = r1*N1*(1 - (N1 + alpha*N2)/K1)
0 = r2*N2*(1 - (N2 + beta*N1)/K2)

Me queda que:
N1 = K1 - alpha*N2
N2 = K2 - beta*N1
'''

# Graficos de las isoclinas de crecimiento poblacional cero (y grafico las trayectorias)

r1, r2 = 0.3, 0.3
N1_0, N2_0 = 10, 10
t0 = 0
tf = np.linspace(1, 100, 100)
h = 0.01

figure1, axis1 = plt.subplots(2, 2)

for i in range(2):
    for j in range(2):
        axis1[i, j].set_xlabel('N1')
        axis1[i, j].set_ylabel('N2')

figure2, axis2 = plt.subplots(2, 2)

for i in range(2):
    for j in range(2):
        axis2[i, j].set_xlabel('Tiempo')
        axis2[i, j].set_ylabel('Población')

# Caso 1 -> Gana la especie 1
alpha = 0.7
beta = 1.2
K1, K2 = 100, 70

N1_values = np.linspace(0, K1, 1000)

S1_isocline = (N1_values - K1)/(- alpha)
S2_isocline = K2 - beta*N1_values

axis1[0, 0].plot(N1_values, S1_isocline, label='Especie 1', color='blue')
axis1[0, 0].plot(N1_values, S2_isocline, label='Especie 2', color='green')
axis1[0, 0].set_ylim(0, K1/alpha + 5)
axis1[0, 0].set_xlim(0, K1 + 5)
axis1[0, 0].legend()
axis1[0, 0].set_title('Gana especie 1')

v1 = runge_kutta4_system(dN1dt, dN2dt, 0, 20, 20, h, n)
axis1[0, 0].arrow(20, 20, v1[0] - 20, v1[1] - 20, linewidth=2 ,head_width=1, head_length=1, fc='black', ec='black')

v2 = runge_kutta4_system(dN1dt, dN2dt, 0, 10, 75, h, n)
axis1[0, 0].arrow(10, 75, v1[0] - 10, v1[1] - 75, linewidth=2 ,head_width=1, head_length=1, fc='black', ec='black')

v3 = runge_kutta4_system(dN1dt, dN2dt, 0, 80, 120, h, n)
axis1[0, 0].arrow(80, 120, v1[0] - 80, v1[1] - 120, linewidth=2 ,head_width=1, head_length=1, fc='black', ec='black')

'''
x, y = np.meshgrid(np.linspace(0, K1/alpha +5, 100), np.linspace(0, K1 + 50, 100)) # para que me cubra todo el grafico
axis1[0, 0].streamplot(x, y, r1*x*(1 - (x + alpha*y)/K1), r2*y*(1 - (y + beta*x)/K2), density=1, color='black')
'''

N1 = []
N2 = []

for t in tf:
    n = int((t-t0)/h)
    N1.append(runge_kutta4_system(dN1dt, dN2dt, t0, N1_0, N2_0, h, n)[0])
    N2.append(runge_kutta4_system(dN1dt, dN2dt, t0, N1_0, N2_0, h, n)[1])

axis2[0, 0].plot(tf, np.array(N1), label='Especie 1', color='blue')
axis2[0, 0].plot(tf, np.array(N2), label='Especie 2', color='green')
axis2[0, 0].legend()
axis2[0, 0].set_title('Gana especie 1')

# Caso 2 -> Gana la especie 2
alpha = 1.5
beta = 0.5
K1, K2 = 70, 60

N1_values = np.linspace(0, K2/beta, 1000) # preguntar si esta bien esto

S1_isocline = (N1_values - K1)/(- alpha)
S2_isocline = K2 - beta*N1_values

axis1[0, 1].plot(N1_values, S1_isocline, label='Especie 1', color='blue')
axis1[0, 1].plot(N1_values, S2_isocline, label='Especie 2', color='green')
axis1[0, 1].set_ylim(0, K2 + 5)
axis1[0, 1].set_xlim(0, K2/beta + 5)
axis1[0, 1].set_title('Gana especie 2')

x, y = np.meshgrid(np.linspace(0, K2/beta + 5, 100), np.linspace(0, K2 + 5, 100))
axis1[0, 1].streamplot(x, y, r1*x*(1 - (x + alpha*y)/K1), r2*y*(1 - (y + beta*x)/K2), density=0.7, color='black')

N1 = []
N2 = []

for t in tf:
    n = int((t-t0)/h)
    N1.append(runge_kutta4_system(dN1dt, dN2dt, t0, N1_0, N2_0, h, n)[0])
    N2.append(runge_kutta4_system(dN1dt, dN2dt, t0, N1_0, N2_0, h, n)[1])

axis2[0, 1].plot(tf, np.array(N1), label='Especie 1', color='blue')
axis2[0, 1].plot(tf, np.array(N2), label='Especie 2', color='green')
axis2[0, 1].set_title('Gana especie 2')

# Caso 3 -> Puede ganar cualquiera
alpha = 1.4
beta = 1.3
K1, K2 = 40, 40

N1_values = np.linspace(0, K1, 1000)

S1_isocline = (N1_values - K1)/(- alpha)
S2_isocline = K2 - beta*N1_values

axis1[1, 0].plot(N1_values, S1_isocline, label='Especie 1', color='blue')
axis1[1, 0].plot(N1_values, S2_isocline, label='Especie 2', color='green')
axis1[1, 0].set_ylim(0, K1/alpha + 5)
axis1[1, 0].set_xlim(0, K1 + 5)
axis1[1, 0].set_title('Puede ganar cualquiera')

x, y = np.meshgrid(np.linspace(0, K1 + 5, 100), np.linspace(0, K1/alpha + 5, 100))
axis1[1, 0].streamplot(x, y, r1*x*(1 - (x + alpha*y)/K1), r2*y*(1 - (y + beta*x)/K2), density=0.7, color='black')

N1 = []
N2 = []

for t in tf:
    n = int((t-t0)/h)
    N1.append(runge_kutta4_system(dN1dt, dN2dt, t0, N1_0, N2_0, h, n)[0])
    N2.append(runge_kutta4_system(dN1dt, dN2dt, t0, N1_0, N2_0, h, n)[1])

axis2[1, 0].plot(tf, np.array(N1), label='Especie 1', color='blue')
axis2[1, 0].plot(tf, np.array(N2), label='Especie 2', color='green')
axis2[1, 0].set_title('Gana cualquiera (Especie 1)')

# Caso 4 -> Coexistencia
alpha = 0.6
beta = 0.6
K1, K2 = 80, 78 # preguntar si esta bien esto

N1_values = np.linspace(0, K2/beta, 1000)

S1_isocline = (N1_values - K1)/(- alpha)
S2_isocline = K2 - beta*N1_values

axis1[1, 1].plot(N1_values, S1_isocline, label='Especie 1', color='blue')
axis1[1, 1].plot(N1_values, S2_isocline, label='Especie 2', color='green')
axis1[1, 1].set_ylim(0, K1/alpha + 5)
axis1[1, 1].set_xlim(0, K2/beta + 5)
axis1[1, 1].set_title('Coexistencia')

x, y = np.meshgrid(np.linspace(0, K2/beta + 5, 100), np.linspace(0, K1/alpha + 5, 100))
axis1[1, 1].streamplot(x, y, r1*x*(1 - (x + alpha*y)/K1), r2*y*(1 - (y + beta*x)/K2), density=0.7, color='black')

N1 = []
N2 = []

for t in tf:
    n = int((t-t0)/h)
    N1.append(runge_kutta4_system(dN1dt, dN2dt, t0, N1_0, N2_0, h, n)[0])
    N2.append(runge_kutta4_system(dN1dt, dN2dt, t0, N1_0, N2_0, h, n)[1])

axis2[1, 1].plot(tf, np.array(N1), label='Especie 1', color='blue')
axis2[1, 1].plot(tf, np.array(N2), label='Especie 2', color='green')
axis2[1, 1].set_title('Coexistencia')

figure1.subplots_adjust(wspace=0.3, hspace=0.3)
figure2.subplots_adjust(wspace=0.3, hspace=0.3)

plt.show()