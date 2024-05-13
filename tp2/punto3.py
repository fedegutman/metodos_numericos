# Defino las ecuaciones del modelo de Predador-Presa de Lotka-Volterra

# Presa
dNdt = lambda t, N, P: r*N - alpha*N*P

# Predador
dPdt = lambda t, N, P: beta*N*P - q*P





