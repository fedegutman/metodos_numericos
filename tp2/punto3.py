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

# Defino las ecuaciones del modelo de Predador-Presa de Lotka-Volterra

# Presa
dNdt = lambda t, N, P: r*N - alpha*N*P
'''
r -> tasa de crecimiento de las presas
N -> numero de individuos (presas)
alpha -> eficiencia de captura
P -> numero de individuos (predadores)
'''

# Predador
dPdt = lambda t, N, P: beta*N*P - q*P
'''
beta -> eficiencia de conversión
N -> numero de individuos (presas)
P -> numero de individuos (predadores)
q -> tasa de mortalidad de los predadores
'''

# Grafico las isoclinas 
'''
0 = r*N - alpha*N*P
0 = beta*N*P - q*P

Me queda que:
P = r/alpha
N = q/beta
'''

# Graficos de las isoclinas de crecimiento poblacional cero

r, alpha = 1.3, 0.5
q, beta = 1.1, 0.3
N = np.linspace(0, 10, 100)

isoclina_presa = np.array([r/alpha for _ in N])
isoclina_predador = np.array([q/beta for _ in N])

plt.plot(N, isoclina_presa, label='Crecimiento poblacional cero de presas', color = 'blue')
plt.plot(isoclina_predador, N, label='Crecimiento poblacional cero de predadores', color = 'red')
plt.xlabel('Población de presas')
plt.ylabel('Población de predadores')
plt.legend()

# Grafico los campos vectoriales
N1, N2 = np.meshgrid(N, N)
dN1dt = r*N1 - alpha*N1*N2
dN2dt = beta*N1*N2 - q*N2
plt.streamplot(N1, N2, dN1dt, dN2dt, density=1, color='black')

plt.show()

# Grafico las trayectorias del sistema

figure, axis = plt.subplots(2, 1)

N0, P0 = 5, 5
t0 = 0
tf = np.linspace(1, 20, 100)
h = 0.01

prey, predator = [], []

for t in tf:
    n = int((t-t0)/h)
    prey.append(runge_kutta4_system(dNdt, dPdt, t0, N0, P0, h, n)[0])
    predator.append(runge_kutta4_system(dNdt, dPdt, t0, N0, P0, h, n)[1])

axis[0].plot(tf, prey, label='Presa', color='blue')
axis[0].set_xlabel('Tiempo')
axis[0].set_ylabel('Población de presas')

axis[1].plot(tf, predator, label='Predador', color='red')
axis[1].set_xlabel('Tiempo')
axis[1].set_ylabel('Población de predadores')

figure.subplots_adjust(hspace=0.3)
plt.show()
