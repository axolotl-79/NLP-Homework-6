import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

eps = np.linspace(0, 1, 11)
A = 1
w = 1 + (3/8)*eps*A**2

def duff(y, e=eps[0]):
    x, v = y
    return np.array([v, -x - e*x**3])

dt = 0.001
t = np.arange(0, 5, dt)

y = np.empty((len(t), 2))
y[0] = A, 0.0

T = np.zeros(len(eps))

i=0
for e in eps:
    for n in range(len(t)-1): 
        y_star = y[n] + 0.5*dt*duff(y[n], e)
        y[n+1] = y[n] + dt*duff(y_star, e)
        if y[n+1, 1] >= 0:
            T[i] = 2*t[n]
            i += 1
            break
i=0

T_approx = 2*np.pi/w

fig, ax = plt.subplots(1, 1, figsize=(5,5))

ax.set_xlabel(r'$\epsilon$')
ax.set_ylabel('T')

ax.plot(eps, T, color='g', linewidth='0.8', marker='x', markeredgecolor='k', label='Numerical')
ax.plot(eps, T_approx, '--', label='Approximation')

ax.legend()

plt.show()