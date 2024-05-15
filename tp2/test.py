import numpy as np
import matplotlib.pyplot as plt

# Define the circle
circle = lambda x, y: x**2 + y**2 - r**2

# Define the radius
r = 1

# Create an array of theta values
theta = np.linspace(0, 2*np.pi, 100)

# Calculate the x and y coordinates of the points on the circle
x = r * np.cos(theta)
y = r * np.sin(theta)

# Plot the circle
plt.figure(figsize=(5,5))
plt.plot(x, y)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()