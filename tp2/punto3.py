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

# Defino las ecuaciones del modelo de Predador-Presa de Lotka-Volterra

# Presa
dNdt = lambda N, P, t: r*N - alpha*N*P
'''
r -> tasa de crecimiento de las presas
N -> numero de individuos (presas)
alpha -> eficiencia de captura
P -> numero de individuos (predadores)
'''

# Predador
dPdt = lambda N, P, t: beta*N*P - q*P
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

# Grafico el diagrama de fases


plt.setx_label('Población de presas')
plt.sety_label('Población de predadores')
V = lambda N, P: beta * N - q*np.log(N) - alpha * P + r*np.log(P)


# axis2.plot(prey, predator, label='Diagrama de fases')

# Lotka-Volterra Extendidas (LVE)

dNdt = lambda t, N, P: r*N*(1 - (N/K)) - alpha*N*P
dPdt = lambda t, N, P: beta*N*P - q*P

# Grafico las isoclinas 
'''
0 = r*N*(1 - (N/K)) - alpha*N*P
0 = beta*N*P - q*P

Me queda que:
P = r/alpha * (1 - N/K)
N = q/beta
'''

r, alpha = 1.3, 0.5
q, beta = 1.1, 0.3
K = 10
N = np.linspace(0, 10, 100)

isoclina_presa = np.array([r/alpha * (1 - n/10) for n in N])
isoclina_predador = np.array([q/beta for _ in N])

plt.plot(N, isoclina_presa, label='Crecimiento poblacional cero de presas', color = 'blue')
plt.plot(isoclina_predador, N, label='Crecimiento poblacional cero de predadores', color = 'red')
plt.xlabel('Población de presas')
plt.ylabel('Población de predadores')
plt.legend()

# Grafico los campos vectoriales
N1, N2 = np.meshgrid(N, N)
plt.streamplot(N1, N2, r*N1*(1 - (N1/K)) - alpha*N1*N2, beta*N1*N2 - q*N2, density=1, color='black')

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