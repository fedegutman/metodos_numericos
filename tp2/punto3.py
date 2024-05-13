import matplotlib.pyplot as plt
import numpy as np

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

figure, axis = plt.subplots(2, 2)

for i in range(2):
    for j in range(2):
        axis[i, j].set_xlabel('N1')
        axis[i, j].set_ylabel('N2')

# Caso 1 ->





