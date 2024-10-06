import casadi
from casadi import Opti, OptiSol
import numpy as np
import data_parsing
import matplotlib.pylab as plt
from tqdm.auto import tqdm

class MHE:

    def __init__(self,
                 N,
                 dt=0.1509433962264151,
                 ground_spring_constant=10,
                 ground_damping_constant=3,
                 force_cost=0.001,
                 rk_error_cost=1000,
                 measured_error_cost=10,
                 force_variance_cost=0.01,
                 force_variance_variance_cost=1):
        self.k = ground_spring_constant
        self.k_p = ground_damping_constant
        self.k_cost = measured_error_cost
        self.k_f = force_cost
        self.k_rk4 = rk_error_cost
        self.k_force_var = force_variance_cost
        self.k_f_v_v = force_variance_variance_cost

        self.xdot = self.make_diff_eq()
        self.h = self.make_h()
        self.update_N(N)
        self.update_dt(dt)

    def read(self, data: data_parsing.SeismicData | np.ndarray):
        if (type(data) == data_parsing.SeismicData):
            self.n = data.time.size
            self.data = data.velocity
        elif (type(data) == np.ndarray):
            self.n = data.size
            self.data = data

    def solve_all(self):
        self.x_est = np.zeros((self.n, 2))

        self.f_est = np.zeros(self.n - 1)
        arrs = []
        lastvars = None
        last_x = None
        last_f = None
        for i in tqdm(range(self.n - self.N + 1)):
            opti = Opti()

            x_vars = [opti.variable(2) for _ in range(self.N)]
            force_vars = [opti.variable() for _ in range(self.N - 1)]
            y_vars = self.data[i:i + self.N]

            c1 = self.cost(*x_vars, *y_vars)
            c2 = sum(f**2 for f in force_vars)
            c3 = sum(
                casadi.sum1((x_vars[j] -
                             self.phi(x_vars[j - 1], force_vars[j - 1]))**2)
                for j in range(1, self.N)) / self.N
            #penalize change in force
            c4 = 0
            for ij in range(self.N - 2):
                c4 += (force_vars[ij] - force_vars[ij + 1])**2
            c5 = 0
            for j in range(self.N-3):
                d1 = force_vars[j] - force_vars[j+1]
                d2 = force_vars[j+1] - force_vars[j+2]
                c5 += (d2 - d1)**2
            c = self.k_cost * c1 + self.k_f * c2 + self.k_rk4 * c3 + self.k_force_var * c4 + self.k_f_v_v * c5
            k_prev = 1000
            if i != 0:
                for j in range(self.N - 2):
                    c += casadi.sum1((x_vars[j] - last_x[j + 1])**
                                     2) * k_prev * (self.N - j) / self.N
                    c += (force_vars[j] -
                          last_f[j + 1])**2 * k_prev * (self.N - j) / self.N
                c += casadi.sum1((x_vars[self.N - 2] - last_x[self.N - 1])**
                                 2) * k_prev * 2 / self.N

            opti.minimize(c)
            opti.solver('ipopt', {"ipopt.print_level": 0, "print_time": False, "ipopt.linear_solver" : "spral"})
            if i != 0:
                opti.set_initial(lastvars)
            sol: OptiSol = opti.solve()
            lastvars = sol.value_variables()
            last_x = [sol.value(x) for x in x_vars]
            last_f = [sol.value(f) for f in force_vars]
            if i == 0:
                for j in range(self.N - 1):
                    self.x_est[j] = sol.value(x_vars[j])
                    self.f_est[j] = sol.value(force_vars[j])
                self.x_est[self.N - 1] = sol.value(x_vars[self.N - 1])
            else:
                self.x_est[i + self.N - 1] = sol.value(x_vars[-1])
                self.f_est[i + self.N - 2] = sol.value(force_vars[-1])
                tqdm.write(str(f"{sol.value(force_vars[-1])}"))
            arrs.append([(opti.value(x), opti.value(f))
                         for x, f in zip(x_vars, force_vars)])
            tqdm.write(str(f"{i}, :"))
            tqdm.write(str(f"\tcost_y: {sol.value(c1)}"))
            tqdm.write(str(f"\tcost_force: {sol.value(c2)}"))
            tqdm.write(str(f"\tcost_rkf: {sol.value(c3)}"))

    def plot(self, fig=None, axis=None, plot=True):
        if fig != axis:
            raise Exception("Need to provide both or neither")
        if fig == None and axis == None:
            fig, ax = plt.subplots(1)

        ax.plot(np.linspace(0, self.dt * self.n, self.n),
                self.x_est[:, 0],
                label="x_tilde")
        ax.plot(np.linspace(0, self.dt * self.n, self.n),
                self.x_est[:, 1],
                label="y_tilde")
        ax.plot(np.linspace(0, self.dt * self.n, self.n - 1),
                self.f_est[:],
                label="force_tilde")

        if plot:
            ax.legend()
            plt.show()
        return fig, ax

    def get_latest_x_est(self):
        return self.x_est

    def update_N(self, N):
        self.N = N
        self.cost = self.make_cost()

    def update_dt(self, dt):
        self.dt = dt
        self.phi = self.make_rk4()

    def make_cost(self):
        x_vars = [casadi.SX.sym("x_" + str(i), 2) for i in range(self.N)]
        y_vars = [casadi.SX.sym("y_" + str(i)) for i in range(self.N)]

        cost = 0
        for x, y in zip(x_vars, y_vars):
            cost += (self.h(x) - y)**2

        return casadi.Function("cost", [*x_vars, *y_vars], [cost])

    def make_diff_eq(self):
        x = casadi.SX.sym('x', 2)
        f = casadi.SX.sym('force')
        return casadi.Function(
            'f', [x, f],
            [casadi.vertcat(x[1], -self.k_p * x[1] - self.k * x[0] + f)],
            ['x', 'f'], ['xdot'])

    def make_h(self):
        x = casadi.SX.sym('x', 2)
        return casadi.Function('h', [x], [x[1]], ['x'], ['y'])

    def make_rk4(self):
        x = casadi.SX.sym('x', 2)
        f = casadi.SX.sym('force')

        k1 = self.xdot(x, f)
        k2 = self.xdot(x + self.dt / 2.0 * k1, f)
        k3 = self.xdot(x + self.dt / 2.0 * k2, f)
        k4 = self.xdot(x + self.dt * k3, f)

        x_1 = x + self.dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return casadi.Function('phi', [x, f], [x_1], ['x', 'force'], ['x1'])
