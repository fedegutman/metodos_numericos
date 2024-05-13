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

# Species 1
dN1dt = lambda t, N1, N2: r1*N1*(1 - (N1 + alpha*N2)/K1)
'alpha -> el efecto que tiene un individuo especie 2 sobre el crecimiento poblacional de la especie 1'
'por eso es que lo multiplico por N2 (la cantidad de individuos de la especie 2)'

#Species 2
dN2dt = lambda t, N1, N2: r2*N2*(1 - (N2 + beta*N1)/K2)
'beta -> el efecto que tiene la especie 1 sobre el crecimiento poblacional de la especie 2'

# Parámetros
# final_N1, final_N2 = runge_kutta4_system(dN1dt, dN2dt, t0, N1_0, N2_0, h, n)

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
K1, K2 = 100, 100

N2_values = np.linspace(0, K2, 1000)
N1_isocline = K1 - alpha*N2_values

N1_values = np.linspace(0, K1, 1000)
N2_isocline = K2 - beta*N1_values

axis[0, 0].plot(N1_isocline, N2_values, label='Especie 1', color='blue')
axis[0, 0].plot(N1_values, N2_isocline, label='Especie 2', color='green')
axis[0, 0].set_ylim(0, K1/alpha + 5)
axis[0, 0].set_xlim(0, K1 + 5)
axis[0, 0].legend()
axis[0, 0].set_title('Gana especie 1')

# Caso 2 -> Gana la especie 2

alpha = 1.5
beta = 0.5
K1, K2 = 100, 100

N2 = np.linspace(0, K2, 1000)
N1_isocline = np.array([(K1 - alpha*n2) for n2 in N2])

N1 = np.linspace(0, K1, 1000)
N2_isocline = np.array([(K2 - beta*n1) for n1 in N1])

axis[0, 1].plot(N1_isocline, N2, label='Especie 1', color='blue')
axis[0, 1].plot(N1, N2_isocline, label='Especie 2', color='green')
axis[0, 1].set_ylim(0, K1/beta + 5)
axis[0, 1].set_xlim(0, K1 + 5)
axis[0, 1].set_title('Gana especie 2')

plt.show()