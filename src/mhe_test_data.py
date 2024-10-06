import casadi
import numpy as np
import matplotlib.pylab as plt
import mhe
import data_parsing

data_reader = data_parsing.DataReader()
data = data_reader.read(0, n_max_subsections=50, n_max_sections=1)

mhe = mhe.MHE(30, dt=data.delta,
              ground_spring_constant=30.93226074688124,
              ground_damping_constant=0.05,
              force_cost=0.000001,
              rk_error_cost=100000000,
              measured_error_cost=1000,
              force_variance_cost=0.0001,
              force_variance_variance_cost=0)

min_v, max_v = data.max_ranges[0]

mhe.read(data.velocity[min_v:max_v])
mhe.solve_all()
fig, ax = mhe.plot(plot=False)
ax[0].plot(np.linspace(0, -data.time[min_v]+data.time[max_v], mhe.n), data.velocity[min_v:max_v], label="x")
ax[0].legend()
ax[1].legend()

plt.show()
