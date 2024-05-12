import numpy as np
import matplotlib.pyplot as plt

K1 = 70
K2 = 50
N1 = np.linspace(0, K1, 100)
N2 = np.linspace(0, K2, 100)

# Case 1: alpha, beta < 1
alpha = 0.3
beta = 0.1
N1_isocline = (K1 - alpha*N2) / (1 - alpha*beta)
N2_isocline = (K2 - beta*N1) / (1 - alpha*beta)
plt.figure()
plt.plot(N2, N1_isocline, label='Species 1 Isocline')
plt.plot(N1, N2_isocline, label='Species 2 Isocline')
plt.legend()
plt.title('Case 1: alpha, beta < 1')
plt.show()

# Case 2: alpha > 1, beta < 1
alpha = 1.2
beta = 0.2
N1_isocline = (K1 - alpha*N2) / (1 - alpha*beta)
N2_isocline = (K2 - beta*N1) / (1 - alpha*beta)
plt.figure()
plt.plot(N2, N1_isocline, label='Species 1 Isocline')
plt.plot(N1, N2_isocline, label='Species 2 Isocline')
plt.legend()
plt.title('Case 2: alpha > 1, beta < 1')
plt.show()

# Case 3: alpha < 1, beta > 1
alpha = 0.2
beta = 1.2
N1_isocline = (K1 - alpha*N2) / (1 - alpha*beta)
N2_isocline = (K2 - beta*N1) / (1 - alpha*beta)
plt.figure()
plt.plot(N2, N1_isocline, label='Species 1 Isocline')
plt.plot(N1, N2_isocline, label='Species 2 Isocline')
plt.legend()
plt.title('Case 3: alpha < 1, beta > 1')
plt.show()

# Case 4: alpha, beta > 1
alpha = 1.2
beta = 1.5
N1_isocline = (K1 - alpha*N2) / (1 - alpha*beta)
N2_isocline = (K2 - beta*N1) / (1 - alpha*beta)
plt.figure()
plt.plot(N2, N1_isocline, label='Species 1 Isocline')
plt.plot(N1, N2_isocline, label='Species 2 Isocline')
plt.legend()
plt.title('Case 4: alpha, beta > 1')
plt.show()