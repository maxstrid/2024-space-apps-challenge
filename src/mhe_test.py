import casadi
import numpy as np
import matplotlib.pylab as plt
import mhe

sigma_x = 0.01
sigma_y = 10
sigma_f = 100

n = 200

mhe = mhe.MHE(50, dt=0.005)


def get_force(t, i):
    return 1000 * np.exp(-t * 5)


x_sim = np.zeros((n, 2))

for i in range(1, n):
    x_sim[i] = (mhe.phi(x_sim[i - 1], get_force(i * mhe.dt, i)) +
                np.array([0, np.random.normal(scale=sigma_x)])).T

y_sim = np.array([mhe.h(x) + np.random.normal(scale=sigma_y) for x in x_sim])
f_sim = [get_force(i * mhe.dt, i) for i in range(1, n)]

mhe.read(y_sim)
mhe.solve_all()
fig, ax = mhe.plot()
ax.plot(np.linspace(0, mhe.dt * n, n), x_sim[:, 0], label="x")
ax.plot(np.linspace(0, mhe.dt * n, n), y_sim[:, 0], label="y")
ax.plot(np.linspace(0, mhe.dt * n, n), x_sim[:, 1], label="vx")
ax.plot(np.linspace(0, mhe.dt * n, n - 1), f_sim, label="force")
# ax.plot([0, 0.5, 0.5, n*dt],[1000,1000,0,0],label="force")
# for i, a in enumerate(arrs):
#     ax.plot(np.linspace(i*dt, (i+mhe.N)*mhe.dt, mhe.N), [g[0] for g in a])
ax.legend()

plt.show()
