# Defino las ecuaciones del modelo de Predador-Presa de Lotka-Volterra

# Prey
dNdt = lambda t, N, P: r*N - alpha*N*P

# Predator
dPdt = lambda t, N, P: beta*N*P - q*P

# Defino las ecuaciones de Lotka-Volterra extendidas (LVE)




