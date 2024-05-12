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

# Caso 1
alpha = 0.2
beta = 0.6
K1 = 100
K2 = 100

# Grafico las isoclinas
N1 = np.linspace(0, K1, 100)
N2 = np.linspace(0, K2, 100)

N1_isocline = K1 - alpha*N2
N2_isocline = K2 - beta*N1

plt.figure()
plt.plot(N2, N1_isocline, label='Species 1 Isocline')
plt.plot(N1, N2_isocline, label='Species 2 Isocline')
plt.legend()
plt.xlabel('N1')
plt.ylabel('N2')
plt.title('Isoclines')
plt.show()

# Caso 2
alpha = 0.6
beta = 0.2
K1 = 100
K2 = 100

N1_isocline = K1 - alpha*N2
N2_isocline = K2 - beta*N1

plt.figure()
plt.plot(N2, N1_isocline, label='Species 1 Isocline')
plt.plot(N1, N2_isocline, label='Species 2 Isocline')
plt.legend()
plt.xlabel('N1')
plt.ylabel('N2')
plt.title('Isoclines')
plt.show()

# Caso 3 las isoclinas se cruzan en un punto de equilibrio
alpha = 0.2 
beta = 0.2
K1 = 100
K2 = 100

N1_isocline = K1 - alpha*N2
N2_isocline = K2 - beta*N1

plt.figure()
plt.plot(N2, N1_isocline, label='Species 1 Isocline')
plt.plot(N1, N2_isocline, label='Species 2 Isocline')
plt.legend()
plt.xlabel('N1')
plt.ylabel('N2')
plt.title('Isoclines')
plt.show()