import matplotlib.pyplot as plt
import numpy as np

def runge_kutta4_system(f1, f2, t0, y0, z0, h, n):
    t = t0
    y = y0
    z = z0
    for i in range(n):
        k1_y = h*f1(t, y, z)
        k1_z = h*f2(t, y, z)
        
        k2_y = h*f1(t + h/2, y + k1_y/2, z + k1_z/2)
        k2_z = h*f2(t + h/2, y + k1_y/2, z + k1_z/2)
        
        k3_y = h*f1(t + h/2, y + k2_y/2, z + k2_z/2)
        k3_z = h*f2(t + h/2, y + k2_y/2, z + k2_z/2)
        
        k4_y = h*f1(t + h, y + k3_y, z + k3_z)
        k4_z = h*f2(t + h, y + k3_y, z + k3_z)
        
        y = y + (k1_y + 2*k2_y + 2*k3_y + k4_y)/6
        z = z + (k1_z + 2*k2_z + 2*k3_z + k4_z)/6
        t = t + h
    return y, z

# Defino las ecuaciones de competencia de Lotka-Volterra

# Especie 1
dN1dt = lambda t, N1, N2: r1*N1*(1 - (N1 + alpha*N2)/K1)
'alpha -> el efecto que tiene un individuo especie 2 sobre el crecimiento poblacional de la especie 1'
'por eso es que lo multiplico por N2 (la cantidad de individuos de la especie 2)'

# Especie 2
dN2dt = lambda t, N2, N1: r2*N2*(1 - (N2 + beta*N1)/K2)
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

# Cuantos parametros usar?

# Grafico las isoclinas 
'''
0 = r1*N1*(1 - (N1 + alpha*N2)/K1)
0 = r2*N2*(1 - (N2 + beta*N1)/K2)

Me queda que:
N1 = K1 - alpha*N2
N2 = K2 - beta*N1
'''

# Graficos de las isoclinas de crecimiento poblacional cero

figure, axis = plt.subplots(2, 2)

for i in range(2):
    for j in range(2):
        axis[i, j].set_xlabel('N1')
        axis[i, j].set_ylabel('N2')

# Caso 1 -> Gana la especie 1
alpha = 0.7
beta = 1.2
K1, K2 = 100, 70

N1_values = np.linspace(0, K1, 1000)

S1_isocline = (N1_values - K1)/(- alpha)
S2_isocline = K2 - beta*N1_values

axis[0, 0].plot(N1_values, S1_isocline, label='Especie 1', color='blue')
axis[0, 0].plot(N1_values, S2_isocline, label='Especie 2', color='green')
axis[0, 0].set_ylim(0, K1/alpha + 5)
axis[0, 0].set_xlim(0, K1 + 5)
axis[0, 0].legend()
axis[0, 0].set_title('Gana especie 1')

# Caso 2 -> Gana la especie 2
alpha = 1.5
beta = 0.5
K1, K2 = 70, 60

N1_values = np.linspace(0, K2/beta, 1000) # preguntar si esta bien esto

S1_isocline = (N1_values - K1)/(- alpha)
S2_isocline = K2 - beta*N1_values

axis[0, 1].plot(N1_values, S1_isocline, label='Especie 1', color='blue')
axis[0, 1].plot(N1_values, S2_isocline, label='Especie 2', color='green')
axis[0, 1].set_ylim(0, K2 + 5)
axis[0, 1].set_xlim(0, K2/beta + 5)
axis[0, 1].set_title('Gana especie 2')

# Caso 3 -> Puede ganar cualquiera
alpha = 1.6
beta = 1.3
K1, K2 = 40, 30

N1_values = np.linspace(0, K1, 1000)

S1_isocline = (N1_values - K1)/(- alpha)
S2_isocline = K2 - beta*N1_values

axis[1, 0].plot(N1_values, S1_isocline, label='Especie 1', color='blue')
axis[1, 0].plot(N1_values, S2_isocline, label='Especie 2', color='green')
axis[1, 0].set_ylim(0, K1/alpha + 5)
axis[1, 0].set_xlim(0, K1 + 5)
axis[1, 0].set_title('Puede ganar cualquiera')

# Caso 4 -> Coexistencia
alpha = 1.2
beta = 0.6
K1, K2 = 80, 60

N1_values = np.linspace(0, K2/beta, 1000)

S1_isocline = (N1_values - K1)/(- alpha)
S2_isocline = K2 - beta*N1_values

axis[1, 1].plot(N1_values, S1_isocline, label='Especie 1', color='blue')
axis[1, 1].plot(N1_values, S2_isocline, label='Especie 2', color='green')
axis[1, 1].set_ylim(0, K1/alpha + 5)
axis[1, 1].set_xlim(0, K2/beta + 5)
axis[1, 1].set_title('Coexistencia')

# Grafico el campo vectorial (NO FUNCA)

N1_values = np.linspace(0, K1, 10) 
N2_values = np.linspace(0, K2, 10) 

for i in N1_values:
    for j in N2_values:
        dN1 = r1*i*(1 - (i + alpha*j)/K1)
        dN2 = r2*j*(1 - (j + beta*i)/K2)
        
        # Normalizar los vectores
        norm = np.hypot(dN1, dN2)
        dN1 /= norm
        dN2 /= norm

        axis[0, 0].quiver(i, j, dN1, dN2, angles='xy', scale_units='xy', color='black')
        axis[0, 1].quiver(i, j, dN1, dN2, angles='xy', scale_units='xy', color='black')
        axis[1, 0].quiver(i, j, dN1, dN2, angles='xy', scale_units='xy', color='black')
        axis[1, 1].quiver(i, j, dN1, dN2, angles='xy', scale_units='xy', color='black')

plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.show()