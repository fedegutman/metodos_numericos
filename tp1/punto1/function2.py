import numpy as np

def f(x1, x2):
    return 0.75*np.exp( (-((10*x1-2)**2)/4) - (-((9*x2-2)**2)/4) )