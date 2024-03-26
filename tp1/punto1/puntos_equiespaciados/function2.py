from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt

def f(x1, x2):
    '''
    x1, x2 ∈ [-1,1]
    '''
    return 0.75*np.exp( (-((10*x1-2)**2)/4) - (((9*x2-2)**2)/4) ) + 0.65*np.exp( (-((9*x1+1)**2)/9) - (((10*x2+1)**2)/2) ) + 0.55*np.exp( (-((9*x1-6)**2)/4) - (((9*x2-3)**2)/4) ) - 0.01*np.exp( (-((9*x1-7)**2)/4) - (((9*x2-3)**2)/4) )

# Grafico utilizando matplotlib
x1 = np.linspace(-1, 1, 100)
x2 = np.linspace(-1, 1, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = f(X1, X2)

# Meshgrid -> For example, if x1 = [1, 2, 3] and x2 = [4, 5, 6], f(x1, x2) will give you the result of applying the function to the pairs (1, 4), (2, 5), and (3, 6). But with X1, X2 = np.meshgrid(x1, x2), you will get the result of applying the function to the pairs (1, 4), (1, 5), (1, 6), (2, 4), (2, 5), (2, 6), (3, 4), (3, 5), and (3, 6).

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection='3d') # 121 es 1 row, 2 columns, primer plot
ax1.plot_surface(X1, X2, Z, cmap='viridis')
ax1.set_title('Original Function')


# Interpolo la funcion f usando grid data (cubic) (ver de usar otras formas)
xi = np.linspace(-1, 1, 15)
yi = np.linspace(-1, 1, 15)
XI, YI = np.meshgrid(xi, yi)

Zi = griddata((X1.flatten(), X2.flatten()), Z.flatten(), (XI, YI), method='cubic') #mmmm chequear esto heavy

ax2 = fig.add_subplot(122, projection='3d')  # 122 means 1 row, 2 columns, second plot
ax2.plot_surface(XI, YI, Zi, cmap='viridis')
ax2.set_title('Interpolated Function')

fig.suptitle("3D Interpolation Function B", fontsize=20)
plt.show()

# Grafico el error de la interpolacion CHEQUEAR
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(111, projection='3d')
ax1.plot_surface(XI, YI, Zi - f(XI, YI), cmap='viridis')
ax1.set_title('Error of the interpolation')
plt.show()