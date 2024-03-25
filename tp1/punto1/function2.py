import numpy as np
import scipy.interpolate as spi
import matplotlib.pyplot as plt

def f(x1, x2):
    '''
    x1, x2 ∈ [-1,1]
    '''
    return 0.75*np.exp( (-((10*x1-2)**2)/4) - (((9*x2-2)**2)/4) ) + 0.65*np.exp( (-((9*x1+1)**2)/9) - (((10*x2+1)**2)/2) ) + 0.55*np.exp( (-((9*x1-6)**2)/4) - (((9*x2-3)**2)/4) ) - 0.01*np.exp( (-((9*x1-7)**2)/4) - (((9*x2-3)**2)/4) )

# grafico utilizando matplotlib
x1 = np.linspace(-1, 1, 100)
x2 = np.linspace(-1, 1, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = f(X1, X2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Z, cmap='viridis')
plt.show()

# interpolo la funcion f usando grid data