from casadi import *
import numpy as np
import matplotlib.pylab as plt


n = 1000

sigma_x = 5
sigma_y = 5

x_sim = np.zeros((n, 2))
dt = 0.005

def make_diffeq():
    k = 10
    k_p = 3

    x = SX.sym('x', 2)
    f = SX.sym('force')
    return Function('f', [x, f], [vertcat(x[1], -k_p * x[1] -k*x[0] + f)], ['x', 'f'], ['x_dot']) #diffeq

xdot = make_diffeq()

def make_h():
    x = SX.sym('x', 2)
    return Function('h', [x], [x[1]], ['x'], ['y']) #state to observed :)

h = make_h()


#to simulate a state...

def make_rk4():
    x = SX.sym('x', 2)
    f = SX.sym('force')
    
    k1 = xdot(x, f)
    k2 = xdot(x+dt/2.0*k1, f)
    k3 = xdot(x+dt/2.0*k2, f)
    k4 = xdot(x+dt*k3, f)

    x_1 = x+dt/6.0*(k1+2*k2+2*k3+k4)
    return Function('phi', [x, f], [x_1], ['x', 'force'], ['x1'])

phi = make_rk4()

def get_force(t, i):
    return 1000 if t < 0.5 else 0

for i in range(1,n):
    x_sim[i] = (phi(x_sim[i - 1], get_force(i * dt, i)) + np.array([0,np.random.normal(scale=sigma_x)])).T

y_sim = np.array([h(x) for x in x_sim]) + np.random.normal(size=(n,1))

N = 10
def make_cost():
    x_vars = [SX.sym("x_"+str(i), 2) for i in range(N)]
    y_vars = [SX.sym("y_"+str(i)) for i in range(N)]
    force_vars = [SX.sym("f_"+ str(i)) for i in range(N)]

    cost = 0
    for x, y in zip(x_vars, y_vars):
        cost += (h(x) - y)**2
    
    return Function("cost", [*x_vars, *force_vars, *y_vars], [cost])

cost = make_cost()

x_est = np.zeros(n)

for i in range(n - N + 1):
    opti = Opti()

    x_vars = [opti.variable(2) for _ in range(N)]
    force_vars = [opti.variable() for _ in range(N)]
    y_vars = y_sim[0][i:i+N]

    opti.minimize(cost(*x_vars, *force_vars, *y_vars))
    #constraints
    for j in range(1, N):
        opti.subject_to(x_vars[j-1] == phi(x_vars[j], force_vars[j]))
    
    opti.solver('ipopt')
    sol: OptiSol = opti.solve()
    x_est[i:i+N] = [*sol.value(x_vars)]

fig, ax = plt.subplots(1)

ax.plot(np.linspace(0, dt*n, n), x_sim[:,0], label="x")
ax.plot(np.linspace(0, dt*n, n), y_sim[:, 0], label="vx/y")
ax.legend()

plt.show()
