from casadi import *
import numpy as np
import matplotlib.pylab as plt

n = 200

sigma_x = 0.01
sigma_y = 10
sigma_f = 100

x_sim = np.zeros((n, 2))
dt = 0.005


def make_diffeq():
    k = 10
    k_p = 3

    x = SX.sym('x', 2)
    f = SX.sym('force')
    return Function('f', [x, f], [vertcat(x[1], -k_p * x[1] - k * x[0] + f)],
                    ['x', 'f'], ['x_dot'])  #diffeq


xdot = make_diffeq()


def make_h():
    x = SX.sym('x', 2)
    return Function('h', [x], [x[1]], ['x'], ['y'])  #state to observed :)


h = make_h()

#to simulate a state...


def make_rk4():
    x = SX.sym('x', 2)
    f = SX.sym('force')

    k1 = xdot(x, f)
    k2 = xdot(x + dt / 2.0 * k1, f)
    k3 = xdot(x + dt / 2.0 * k2, f)
    k4 = xdot(x + dt * k3, f)

    x_1 = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return Function('phi', [x, f], [x_1], ['x', 'force'], ['x1'])


phi = make_rk4()


def get_force(t, i):
    return 1000 * exp(-t * 5)


for i in range(1, n):
    x_sim[i] = (phi(x_sim[i - 1], get_force(i * dt, i)) +
                np.array([0, np.random.normal(scale=sigma_x)])).T

y_sim = np.array([h(x) + np.random.normal(scale=sigma_y) for x in x_sim])
f_sim = [get_force(i * dt, i) for i in range(1, n)]

N = 50


def make_cost():
    x_vars = [SX.sym("x_" + str(i), 2) for i in range(N)]
    y_vars = [SX.sym("y_" + str(i)) for i in range(N)]

    cost = 0
    for x, y in zip(x_vars, y_vars):
        cost += (h(x) - y)**2

    return Function("cost", [*x_vars, *y_vars], [cost])


cost = make_cost()

x_est = np.zeros((n, 2))
f_est = np.zeros(n - 1)
arrs = []
lastvars = None
last_x = None
last_f = None
for i in range(n - N + 1):
    opti = Opti()

    x_vars = [opti.variable(2) for _ in range(N)]
    force_vars = [opti.variable() for _ in range(N - 1)]
    y_vars = y_sim[i:i + N, 0]

    c1 = cost(*x_vars, *y_vars) / sigma_y
    k_f = 0.001
    k_rk4 = 1000
    c2 = k_f * sum(f**2 for f in force_vars) / sigma_f
    c3 = k_rk4 * sum(
        sum1((x_vars[j] - phi(x_vars[j - 1], force_vars[j - 1]))**2)
        for j in range(1, N)) / sigma_x / N
    #penalize change in force
    c4 = 0
    for ij in range(N - 2):
        c4 += (force_vars[ij] - force_vars[ij + 1])**2
    #TODO: penalize change in change in force
    k_prev = 0.1
    if i != 0:
        for j in range(N - 2):
            c += sum1((x_vars[j] - last_x[j + 1])**2) * k_prev * (N - j) / N
            c += (force_vars[j] - last_f[j + 1])**2 * k_prev * (N - j) / N
        c += sum1((x_vars[N - 2] - last_x[N - 1])**2) * k_prev * 2 / N
    c5 = 0
    for j in range(N-3):
        d1 = force_vars[j] - force_vars[j+1]
        d2 = force_vars[j+1] - force_vars[j+2]
        c5 += (d2 - d1)**2
    else:
        opti.subject_to(x_vars[0][0] == 0)
        opti.subject_to(x_vars[0][1] == 0)

    c = c1 * 10 + c2 + c3 + c4 * 0.01 + c5
    opti.minimize(c)
    opti.solver('ipopt', {
        "ipopt.print_level": 0,
        "print_time": False,
        'ipopt.max_iter': 10000
    })
    if i != 0:
        opti.set_initial(lastvars)
    sol: OptiSol = opti.solve()
    lastvars = sol.value_variables()
    last_x = [sol.value(x) for x in x_vars]
    last_f = [sol.value(f) for f in force_vars]
    if i == 0:
        for j in range(N - 1):
            x_est[j] = sol.value(x_vars[j])
            f_est[j] = sol.value(force_vars[j])
        x_est[N - 1] = sol.value(x_vars[N - 1])
    else:
        x_est[i + N - 1] = sol.value(x_vars[-1])
        f_est[i + N - 2] = sol.value(force_vars[-1])
        print(sol.value(force_vars[-1]))
    arrs.append([(opti.value(x), opti.value(f))
                 for x, f in zip(x_vars, force_vars)])
    print(i, ":")
    print("\tcost_y:", sol.value(c1))
    print("\tcost_force:", sol.value(c2))
    print("\tcost_rkf:", sol.value(c3))

fig, ax = plt.subplots(1)
ax.plot(np.linspace(0, dt * n, n), x_sim[:, 0], label="x")
ax.plot(np.linspace(0, dt * n, n), y_sim[:, 0], label="y")
ax.plot(np.linspace(0, dt * n, n), x_sim[:, 1], label="vx")
ax.plot(np.linspace(0, dt * n, n), x_est[:, 0], label="x_tild")
ax.plot(np.linspace(0, dt * n, n), x_est[:, 1], label="y_tild")
ax.plot(np.linspace(0, dt * n, n - 1), f_est[:], label="FORCE")
ax.plot(np.linspace(0, dt * n, n - 1), f_sim, label="force")
# ax.plot([0, 0.5, 0.5, n*dt],[1000,1000,0,0],label="force")
# for i, a in enumerate(arrs):
#     ax.plot(np.linspace(i*dt, (i+N)*dt, N), [g[0] for g in a])
ax.legend()

plt.show()
