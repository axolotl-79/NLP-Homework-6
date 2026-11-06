import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

eps = 1
A = 1
w = 1 + (3/8)*eps*A**2


def duff(y):
    x, v = y
    return np.array([v, -x - eps*x**3])

dt = 0.1
t = np.arange(0, 20, dt)

y = np.empty((len(t), 2))
y[0] = A, 0.0

for n in range(len(t)-1): 
    y_star = y[n] + 0.5*dt*duff(y[n])
    y[n+1] = y[n] + dt*duff(y_star)


x = A*np.cos(w*t) - eps*(A**2)/32*(np.cos(3*w*t)-np.cos(w*t))
fig, ax = plt.subplots(1, 1, figsize=(6,6))

ax.set_xlabel('t')
ax.set_ylabel('x')

ax.plot(t, x,linewidth='0.8', label='Approximation')
ax.plot(t, y[:,0], 
        marker='.', markersize='3', markeredgecolor='k', markerfacecolor='k',
        linewidth='0', label='Numerical')
ax.legend()
plt.show()