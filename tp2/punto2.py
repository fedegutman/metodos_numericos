def runge_kutta4(f, t0, y0, h, n):
    t = t0
    y = y0
    for i in range(n):
        k1 = h*f(t, y)
        k2 = h*f(t + h/2, y + k1/2)
        k3 = h*f(t + h/2, y + k2/2)
        k4 = h*f(t + h, y + k3)
        y = y + (k1 + 2*k2 + 2*k3 + k4)/6
        t = t + h
    return y

# Defino las ecuaciones de competencia de Lotka-Volterra
dN1dt = lambda t, N1, N2: r1*N1*(1 - (N1 + alpha*N2)/K1)

dN2dt = lambda t, N1, N2: r2*N2*(1 - (N2 + beta*N1)/K2)

# Parámetros