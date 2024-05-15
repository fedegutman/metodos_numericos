import numpy as np
import matplotlib.pyplot as plt

# Define parameters
beta = 0.5
q = 0.1
alpha = 0.3
r = 0.2

# Define the function V
def V(N, P):
    return beta * N - q * np.log(N) - alpha * P + r * np.log(P)

# Define the range of N and P values
N_values = np.linspace(0.1, 5, 100)
P_values = np.linspace(0.1, 5, 100)

# Create a grid of N and P values
N, P = np.meshgrid(N_values, P_values)

# Calculate the values of V for each point in the grid
Z = V(N, P)

# Plot the contour plot
plt.figure(figsize=(8, 6))
contour = plt.contour(N, P, Z, levels=20, cmap='RdGy')
plt.colorbar(contour, label='V(N, P)')
plt.xlabel('N (Prey)')
plt.ylabel('P (Predator)')
plt.title('Phase Diagram - Lotka-Volterra Predator-Prey Model')
plt.grid(True)
plt.xlim(0, 5)  # Adjust limits to ensure non-negative logarithm values
plt.ylim(0, 5)  # Adjust limits to ensure non-negative logarithm values
plt.show()
