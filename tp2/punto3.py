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

def runge_kutta4_system_phaseplot(f1, f2, t0, x0, y0, h, n):
    t = t0
    x = x0
    y = y0
    xs = [x0]
    ys = [y0]
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
        xs.append(x)
        ys.append(y)
    return xs, ys

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

N1, N2 = np.meshgrid(np.linspace(0.1, 10, 15), np.linspace(0.1, 10, 15))
magnitude = np.sqrt((r*N1 - alpha*N1*N2)**2 + (beta*N1*N2 - q*N2)**2)
normalized_r1 = (r*N1 - alpha*N1*N2) / magnitude
normalized_r2 = (beta*N1*N2 - q*N2) / magnitude
plt.quiver(N1, N2, normalized_r1, normalized_r2, color='black',)

plt.show()

figure, axis = plt.subplots(2, 1)

N0, P0 = 5, 5
t0 = 0
tf = np.linspace(1, 40, 200)
h = 0.01

prey, predator = [], []
parameters = [(0.3, 0.3, 1.3, 0.9, 15), (0.4, 0.2, 1.8, 0.7, 20), (0.1, 0.2, 0.8, 0.8, 25)]
colors = ['m', 'b', 'c']
linestyles = ['-', '--', ':']

for i in range(3):
    alpha, beta, r, q, K = parameters[i]
    prey, predator = [], []
    for t in tf:
        n = int((t-t0)/h)
        prey_new, predator_new = runge_kutta4_system(dNdt, dPdt, t0, N0, P0, h, n)
        prey.append(prey_new)
        predator.append(predator_new)
    axis[0].plot(tf, prey, label=f'alpha = {alpha}, beta = {beta}, r = {r}, q = {q}, K = {K}', color=colors[i], linestyle=linestyles[i])
    axis[1].plot(tf, predator, label=f'alpha = {alpha}, beta = {beta}, r = {r}, q = {q}, K = {K}', color=colors[i], linestyle=linestyles[i])

# axis[0].plot(tf, prey, label='Presa', color='blue')
axis[0].set_xlabel('Tiempo')
axis[0].set_ylabel('Población de presas')
axis[0].set_ylim(0, 14)

# axis[1].plot(tf, predator, label='Predador', color='red')
axis[1].set_xlabel('Tiempo')
axis[1].set_ylabel('Población de predadores')
axis[1].set_ylim(0, 14)
axis[0].legend()

figure.subplots_adjust(hspace=0.3)
plt.show()

# Grafico el diagrama de fases

alpha = 0.9
beta = 0.8
q = 1.2
r = 1.1
condiciones_iniciales = [(2, 2), (3, 2), (4, 2), (5, 2), (2, 3), (2, 4), (2, 5)]

t0 = 0
tf = 10
h = 0.01
n = int((tf - t0) / h)

for i in range(7):
    N0, P0 = condiciones_iniciales[i]
    prey, predator = runge_kutta4_system_phaseplot(dNdt, dPdt, t0, N0, P0, h, n)
    plt.plot(prey, predator, label=f'N0, P0 = {N0}, {P0}')

plt.xlabel('Población de presas')
plt.ylabel('Población de predadores')
plt.legend()

'''
N1, N2 = np.meshgrid(np.linspace(0.1, 10, 15), np.linspace(0.1, 10, 15))
magnitude = np.sqrt((r*N1 - alpha*N1*N2)**2 + (beta*N1*N2 - q*N2)**2)
normalized_r1 = (r*N1 - alpha*N1*N2) / magnitude
normalized_r2 = (beta*N1*N2 - q*N2) / magnitude
plt.quiver(N1, N2, normalized_r1, normalized_r2, color='black',)

'''
plt.show()


# Lotka-Volterra Extendidas (LVE)

dNdt = lambda N, P, t: r*N*(1 - (N/K)) - alpha*N*P
dPdt = lambda N, P, t: beta*N*P - q*P

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
N1, N2 = np.meshgrid(np.linspace(0.1, 10, 15), np.linspace(0.1, 10, 15))
magnitude = np.sqrt((r*N1*(1 - N1/K) - alpha*N1*N2)**2 + (beta*N1*N2 - q*N2)**2)
normalized_r1 = (r*N1*(1 - N1/K) - alpha*N1*N2) / magnitude
normalized_r2 = (beta*N1*N2 - q*N2) / magnitude
plt.quiver(N1, N2, normalized_r1, normalized_r2, color='black',)

plt.show()

# Grafico las trayectorias del sistema
figure, axis = plt.subplots(2, 1)

N0, P0 = 6, 6
t0 = 0
tf = np.linspace(1, 40, 100)
h = 0.01

prey, predator = [], []

for i in range(3):
    alpha, beta, r, q, K = parameters[i]
    prey, predator = [], []
    for t in tf:
        n = int((t-t0)/h)
        prey_new, predator_new = runge_kutta4_system(dNdt, dPdt, t0, N0, P0, h, n)
        prey.append(prey_new)
        predator.append(predator_new)
    axis[0].plot(tf, prey, label=f'alpha = {alpha}, beta = {beta}, r = {r}, q = {q}, K = {K}', color=colors[i], linestyle=linestyles[i])
    axis[1].plot(tf, predator, label=f'alpha = {alpha}, beta = {beta}, r = {r}, q = {q}, K = {K}', color=colors[i], linestyle=linestyles[i])

axis[0].set_xlabel('Tiempo')
axis[0].set_ylabel('Población de presas')
axis[0].set_ylim(0, 10)
axis[0].legend()

axis[1].set_xlabel('Tiempo')
axis[1].set_ylabel('Población de predadores')
axis[1].set_ylim(0, 10)

figure.subplots_adjust(hspace=0.3)
plt.show()

# Grafico el diagrama de fases ESTO A CHEQUEEAAAAR

alpha = 0.9
beta = 0.8
q = 1.2
r = 1.1
condiciones_iniciales = [(2, 2), (3, 2), (4, 2), (5, 2), (2, 3), (2, 4), (2, 5)]
K = 10
N0 = 2
P0 = 2

t0 = 0
tf = 30
h = 0.01
n = int((tf - t0) / h)

for i in range(7):
    N0, P0 = condiciones_iniciales[i]
    prey, predator = runge_kutta4_system_phaseplot(dNdt, dPdt, t0, N0, P0, h, n)
    plt.plot(prey, predator, label=f'N0, P0 = {N0}, {P0}')


plt.xlabel('Población de presas')
plt.ylabel('Población de predadores')
plt.legend()

'''
N1, N2 = np.meshgrid(np.linspace(0.1, 10, 15), np.linspace(0.1, 10, 15))
magnitude = np.sqrt((r*N1*(1 - N1/K) - alpha*N1*N2)**2 + (beta*N1*N2 - q*N2)**2)
normalized_r1 = (r*N1*(1 - N1/K) - alpha*N1*N2) / magnitude
normalized_r2 = (beta*N1*N2 - q*N2) / magnitude
plt.quiver(N1, N2, normalized_r1, normalized_r2, color='black',)
'''

plt.show()